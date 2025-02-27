def getDataStore (container_name):
    """Return a data store blob for loading blob data given a container name"""

    # Authentication package
    from azure.identity import DefaultAzureCredential

    # Handle to the workspace
    from azure.ai.ml import MLClient
    from azureml.core import Workspace, Datastore

    # replace subscription id 
    subscription_id = 'YOUR_SUBSCRIPTION_ID'
    resource_group = 'YOUR_RESOURCE_GROUP'
    workspace_name = 'YOUR_WORKSPACE_NAME'

    credential = DefaultAzureCredential()

    # Get a handle to the workspace. You can find the info on the workspace tab on ml.azure.com

    # This ML Client stores all of our private information
    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )

    # create a workspace
    workspace = Workspace(subscription_id, resource_group, workspace_name)

    # get data from data container into our workspace directly 
    blob_datastore = Datastore.get(workspace, container_name)

    return blob_datastore

def getReadWriteDataStore():
    """Return read-write data store blob for loading blob data"""

    # Authentication package
    from azure.identity import DefaultAzureCredential

    # Handle to the workspace
    from azure.ai.ml import MLClient
    from azureml.core import Workspace, Datastore

    # replace subscription id 
    subscription_id = 'YOUR_SUBSCRIPTION_ID'
    resource_group = 'YOUR_RESOURCE_GROUP'
    workspace_name = 'YOUR_WORKSPACE_NAME'

    credential = DefaultAzureCredential()

    # Get a handle to the workspace. You can find the info on the workspace tab on ml.azure.com

    # This ML Client stores all of our private information
    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )

    # create a workspace
    workspace = Workspace(subscription_id, resource_group, workspace_name)

    # get data from data container into our workspace directly 
    blob_datastore = Datastore.get(workspace, 'YOUR_DATASTORE_NAME')

    return blob_datastore


def load_data_file (file, blob_datastore, path, debug = False):
    """Load a data file and return the Pandas data frame

    Parameters:
    file (string): the file name without a trailing extension suffix
    blob_datastore: the custom data store object 
    path (string): the path to the file, with a trailing '/'
    debug (boolean): prints the number of lines in the data frame

    Returns:
    A Pandas dataframe
    """

    # Handle to the workspace
    from azureml.core import Dataset

    dataset = Dataset.Tabular.from_delimited_files(
        path = (blob_datastore, path + file + '.csv')
    )
    
    df = dataset.to_pandas_dataframe()

    if debug: print(f'{len(df)} result rows in file [' + file + ']')

    return df
