#!/usr/bin/python3.7

import aiohttp
from aiohttp import web, WSCloseCode
import asyncio
import datetime
import tensorflow as tf
import numpy as np
import cv2
import os

#model stuff
model_path='./model_letters_192x192_dense201_hyper_85.14_1loss.h5'
image_side=192
classes = ("A","B","C","D","E","F","G","H","I","K","L","M","N","O","P","Q","R","S", "T", "U", "V", "W", "X", "Y") # since our classes have just 1 character this can be also "ABCDEFGHIKLMNOPQRSTUWXY" but we prefer to use a tuple

try:
    model=tf.keras.models.load_model(model_path)
except:
    ## print original exception
    import traceback
    print(traceback.format_exc())
    print("If you get error \"expected 2 variables, but received 0 variables during loading.\" try to change model path from .keras to .h5")
    exit()

#mutex for model
model_mutex = asyncio.Lock()

#web
HTML_DIR = "html"

#xx/example_imgs/A.jpg
async def img_handler(request):
    imagesFolder = HTML_DIR + os.sep + "example_imgs"
    try:
        img = request.match_info['img']

        #remove extension
        img = img.split(".")[0]

        if img in classes:
            with open(imagesFolder + os.sep + img + ".webp", "rb") as f:
                content = f.read()
                return web.Response(body=content, content_type="image/webp")
        else:
            return web.Response(text='File not found')
    except:
        return web.Response(text='File not found')


async def http_handler(request):
    try:
        with open(HTML_DIR + "/index.html", "rb") as f:
            content = f.read()
            return web.Response(body=content, content_type="text/html")
    except FileNotFoundError:
        return web.Response(text='File not found')


async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    async for msg in ws:
        if msg.type == aiohttp.WSMsgType.TEXT:
            if msg.data == 'close':
                await ws.close()
            else:
                await ws.send_str('some websocket message payload')
        elif msg.type == aiohttp.WSMsgType.ERROR:
            print('ws connection closed with exception %s' % ws.exception())
        elif msg.type == aiohttp.WSMsgType.BINARY:
            print("img to process received")

            # get width and height from msg.data
            frame = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)

            # if image is not square crop it keeping the center
            if frame.shape[0] != frame.shape[1]:
                if frame.shape[0] > frame.shape[1]:
                    #crop y
                    y = frame.shape[0] - frame.shape[1]
                    y = int(y / 2)
                    frame = frame[y:y+frame.shape[1], 0:frame.shape[1]]
                else:
                    #crop x
                    x = frame.shape[1] - frame.shape[0]
                    x = int(x / 2)
                    frame = frame[0:frame.shape[0], x:x+frame.shape[0]]

            frameCopy = frame.copy()
            # if frame shape is not the same of image_side, resize it
            if frame.shape[0] != image_side or frame.shape[1] != image_side:
                # resize the frame
                frame = cv2.resize(frame, (image_side, image_side), interpolation = cv2.INTER_AREA)
                frameResized = frame.copy()

            frame = frame / 255.0
            frame = frame.reshape(-1, image_side, image_side, 3)

            #lock model
            await model_mutex.acquire()

            pred = model.predict(frame)

            #unlock model
            model_mutex.release()

            # index of predicted class
            index = np.argmax(pred)
            percent = str(round(pred[0][index] * 100, 2))

            # if prediction is < 20% send x to client to ignore result
            if float(percent) < 20:
                await ws.send_str("x")
            else:
                await ws.send_str(classes[index]+ " " + percent + "%")

    return ws


def create_runner():
    app = web.Application()
    app.add_routes([
        
        web.get('/',   http_handler),
        web.get('/ws', websocket_handler),
        web.get('/example_imgs/{img}', img_handler),
        #match also when there is not the last /
        web.get('/{path:.*}', http_handler)

    ])
    return web.AppRunner(app)


async def start_server(host="0.0.0.0", port=8000):
    runner = create_runner()
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    print("Server started at http://localhost:8000")
    loop.run_until_complete(start_server())
    loop.run_forever()