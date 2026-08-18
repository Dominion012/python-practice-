from pydantic import BaseModel
from fastapi import FastAPI, BackgroundTasks
import uvicorn
import time 


app = FastAPI()

class Input(BaseModel):
    item: str
    quantity: int
def order_log(item:str, quantity:int):
    time.sleep(2)
    message = f"Logged: An order of {item} and quantity of {quantity}"
    with open ("orders.txt", "a") as f:
        f.write (message + "\n")  
    
        return {"message" : message}

def ware_note(item:str, quantity: int):
    time.sleep(2)
    print(f"Prepare {quantity}units of {item}")

def receipt(item:str):
    time.sleep(2)
    print(f"Receipt sent for {item}")


@app.post("/order_log")
def order_process(request:Input, backgroundtask:BackgroundTasks):
    backgroundtask.add_task(order_log, request.item, request.quantity)
    backgroundtask.add_task(ware_note,request.item, request.quantity)
    backgroundtask.add_task(receipt, request.item)
    return {"message" : f"Order of {request.item} of {request.quantity} units has been processed"}


if __name__ == "__main__":
    uvicorn.run(app, host= "0.0.0.0", port = 8009)