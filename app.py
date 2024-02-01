from typing import Optional
from dotmap import DotMap
import chainlit as cl
import jwt
import logging
from http.cookies import SimpleCookie


@cl.on_message
async def main(message: cl.Message):
    # Your custom logic goes here...

    # Send a response back to the user
    await cl.Message(
        content=f"Received: {message.content}",
    ).send()

@cl.header_auth_callback
def header_auth_callback(headers: dict) -> Optional[cl.User]:
    # NOTE: The authentication requires the chatbot to be running on a subdomain of the same domain used by the lti tool.
    rawdata = headers.get('cookie')
    if rawdata:
        try:
            cookie = SimpleCookie()
            cookie.load(rawdata)
            cookies = {k: v.value for k, v in cookie.items()}
            token = cookies.get('token')
        except:
            return None
    else:
        return None

    if token:
        try:
            logging.debug("Got token.")
            # Read rsa public key
            file = open('rs256.rsa.pub', mode='r')
            key = file.read()
            file.close()
            logging.debug("Attempting jwt decode.")
            payload = jwt.decode(token, key, algorithms="RS256")
            logging.debug("Successfull decode")
            payload = DotMap(payload)
            adminRoleKey = 'http://purl.imsglobal.org/vocab/lis/v2/institution/person#Administrator'
            isAdmin = adminRoleKey in payload.platformContext.roles
            role = 'admin' if isAdmin else 'student'
            return cl.User(identifier=payload.user, metadata={"role": role, "provider": "header"})
        except:
            logging.error("JWT decode failed")
            return None
    else:
        return None
