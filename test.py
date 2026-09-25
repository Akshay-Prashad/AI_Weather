import os, eumdac
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()

token = eumdac.AccessToken((os.getenv("EUMETSAT_CONSUMER_KEY"), os.getenv("EUMETSAT_CONSUMER_SECRET")))
ds = eumdac.DataStore(token)

end = datetime.utcnow() - timedelta(days=2)
start = end - timedelta(hours=3)

for col in ds.collections:
    title = str(col.title)
    if "level 1" not in title.lower():
        continue
    try:
        prods = list(col.search(dtstart=start, dtend=end))
        if not prods:
            print(f"[no data]  {col}  {title}")
            continue
        with prods[0].open() as f:
            f.read(1024)
        print(f"[OK]       {col}  {title}")
    except Exception as e:
        status = "403" if "403" in str(e) else type(e).__name__
        print(f"[{status}]  {col}  {title}")
