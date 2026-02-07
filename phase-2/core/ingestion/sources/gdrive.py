

from io import BytesIO
from typing import Any, Iterator
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.service_account import Credentials
from core.ingestion.ingest import IngestionService

GOOGLE_EXPORT_TARGETS: dict[str, tuple(str, str)] = {
    "application/vnd.google-apps.document": (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".docx",
    ),
    "application/vnd.google-apps.spreadsheet": (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xlsx",
    ),
    "application/vnd.google-apps.presentation": (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".pptx",
    )
}


GDRIVE_SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly"
]

class GDriveIngestor:

    def __init__(self, service_account_json: dict[str, Any], ingestion: IngestionService) -> None:
        creds = Credentials.from_service_account_info(service_account_json, scopes=GDRIVE_SCOPES)
        self.drive = build("drive", "v3", credentials=creds)
        self.ingestion = ingestion

    def list_files(self, folder_id:str) -> Iterator[dict[str, Any]]:
        page_token = None
        while True:
            response = self.drive.files().list(
                q=f'{folder_id} in parents and trashed = false',
                fields="nextPageToken, files(id, name, mimeType, modifiedTime, webViewLink)",
                pageToken=page_token
            ).execute()
            for f in response.get("files", []):
                yield f
            page_token = response.get("nextPageToken")
            if not page_token:
                break

    def _download_or_export(self, file: dict[str, Any]) -> tuple[bytes, str]:

        # Export native google files like sheets,docs,slide and download
        # Else download file directly

        mime_type = file.get("mimeType", "")
        name = file.get("name", "unknown")

        if mime_type in GOOGLE_EXPORT_TARGETS:
            export_mime, ext = GOOGLE_EXPORT_TARGETS[mime_type]
            request = self.drive.files().export_media(fileId=file["id"], mimeType=export_mime)
            final_name = f"{name}{ext}"
        else:
            request = self.drive.files().get_media(fileId=file["id"])
            final_name = name
        
        file_handler = BytesIO()
        downloader = MediaIoBaseDownload(file_handler, request)
        done = False
        
        while not done:
            _, done = downloader.next_chunk()
        return file_handler.getvalue(), final_name
    

    async def ingest_folder(self, folder_id: str) -> list[dict[str, Any]]:
        results = []
        for file in self.list_files(folder_id):
            # need to do this asyncronously
            data, final_name = self._download_or_export(file)
            metadata = {
                "source": "g_drive",
                "webViewLink": file.get("webViewLink"),
                "modifiedTime": file.get("modifiedTime")
            }

            result = await self.ingestion.ingest(final_name, data, metadata)
            results.append({"name": final_name, "result": result})
            return results

    


    
