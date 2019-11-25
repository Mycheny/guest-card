import asyncio
import websockets
import json


async def test_ws_quote():
    async with websockets.connect('ws://127.0.0.1:10000/websocket') as websocket:
    # async with websockets.connect('ws://192.168.160.219:10000/websocket') as websocket:
        # req = {"protocol": "history_req", 'code': 'XAGODS', 'type': 'MINUTE', 'start_pos': '0', 'pos_num': '10'}
        req = {}
        await websocket.send(json.dumps(req))

        while True:
            quote = await websocket.recv()
            print(quote)


asyncio.get_event_loop().run_until_complete(test_ws_quote())
print("dasda")