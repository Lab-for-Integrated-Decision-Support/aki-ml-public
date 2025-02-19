from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

class AzureBlobStorageConnection:

    def __init__(self, storage_url, credential): 
        
        self.storage_url = storage_url
        self.credential = credential
        self.blob_service_client = None

    def __enter__(self):
        """
        open connections using existing credentials
        """

        # Create the BlobServiceClient object using sas_tokens or storage key to connect and manipulate blob storage
        self.blob_service_client = BlobServiceClient(self.storage_url, credential=self.credential)
        
        print("open connection")

        return self.blob_service_client

    def __exit__(self, exc_type, exc_value, traceback):
        """
        close existing connections
        """
        self.blob_service_client.close()
        print("close connection")
