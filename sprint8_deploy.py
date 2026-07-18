import modal

app = modal.App("homeiq-demo")

@app.function()
def hello_homeiq():
    return "Hello from HomeIQ, running in the cloud!"

@app.local_entrypoint()
def main():
    result = hello_homeiq.remote()
    print(result)