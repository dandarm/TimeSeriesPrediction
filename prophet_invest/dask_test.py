from dask.distributed import Client,LocalCluster
#client = Client("127.0.0.1:8786")  # set up local cluster on your laptop

#cluster = LocalCluster(dashboard_address='127.0.0.1:8782')

client = Client()

#print(client)