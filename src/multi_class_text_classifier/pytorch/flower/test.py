from client import PyTorchClient

client = PyTorchClient(country="Germany")
parameters = client.get_parameters(config=None)
client.set_parameters(parameters=parameters)
client.fit(parameters=parameters, config=None)