import requests


class SlackMessenger:
    def __init__(self, token, target, name):
        self.token = token
        self.channel = target
        self.name = name

    def message(self, message):

        data = {
            "token": self.token,
            "channel": self.channel,
            "as_user": True,
            "text": f"{self.name}: " + message,
        }

        r = requests.post(url="https://slack.com/api/chat.postMessage", data=data)

        return r.status_code
