# docker run -p 7080:7080

# headers = {"Content-Type": "application/json"}
# >>> r = requests.post('http://0.0.0.0:7080/predictions/kaveri', headers=headers, data={'instances':[{'json':'meow'}]})

import json

import requests

headers = {"Content-Type": "application/json"}
sample = json.load(open("./data/kaveri_sample_2020_10_01.json"))

call_sample = {"instances": [sample]}

url = "https://h2ox-ai-kdjsv6lupq-ez.a.run.app/predictions/kaveri"
# url = 'http://0.0.0.0:7080/predictions/kaveri'

r = requests.post(url, headers=headers, data=json.dumps(call_sample).encode("utf-8"))

print("STATUS CODE")
print(r.status_code)
print("TEXT")
print(r.text)
