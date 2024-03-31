import time
import requests
import random
# from tqdm import tqdm

headers = {
    'Content-Type': 'application/json',
}

sentence = """The Second War of Scottish Independence broke out in 1332 when Edward Balliol led an English-backed invasion of Scotland. Balliol, the son of former Scottish king John Balliol, was attempting to make good his claim to the Scottish throne. He was opposed by Scots loyal to the occupant of the throne, eight-year-old David II. At the Battle of Dupplin Moor Balliol's force defeated a Scottish army ten times their size and Balliol was crowned king. Within three months David's partisans had regrouped and forced Balliol out of Scotland. He appealed to the English king, Edward III, who invaded Scotland in 1333 and besieged the important trading town of Berwick. A large Scottish army attempted to relieve it but was heavily defeated at the Battle of Halidon Hill. Balliol established his authority over most of Scotland, ceded to England the eight counties of south-east Scotland and did homage to Edward for the rest of the country as a fief. As allies of Scotland via the Auld Alliance, the French were unhappy about an English expansion into Scotland and so covertly supported and financed David's loyalists. Balliol's allies fell out among themselves and he lost control of most of Scotland again by late 1334. In early 1335 the French attempted to broker a peace. However, the Scots were unable to agree a position and Edward prevaricated while building a large army. He invaded in July and again overran most of Scotland. Tensions with France increased. Further French-sponsored peace talks failed in 1336 and in May 1337 the French king, Philip VI, engineered a clear break between France and England, starting the Hundred Years' War. The Anglo-Scottish war became a subsidiary theatre of this larger Anglo-French war. Edward sent what troops he could spare to Scotland, in spite of which the English slowly lost ground in Scotland as they were forced to focus on the French theatre. Achieving his majority David returned to Scotland from France in 1341 and by 1342 the English had been cleared from north of the border."""

start = time.time()

json_data = {
    'inputs': [
        {
            'name': 'text',
            'shape': [1],
            'datatype': 'BYTES',
            'data': [sentence],
        },
    ],
}

response = requests.post(
    'http://localhost:8000/v2/models/summarization_bart_large_cnn/versions/1/infer',
    headers=headers,
    json=json_data,
)
if "error" in response.json():
    print(response.json())
print(response.json())

print("Total time : ", time.time() - start)