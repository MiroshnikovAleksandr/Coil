import tomli
with open('parameters.toml', 'rb') as toml:
    parameters = tomli.load(toml)

print(parameters)
