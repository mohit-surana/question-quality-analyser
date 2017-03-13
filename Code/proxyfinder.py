# Invocation: python3 proxy_finder.py 1000 500 (mines 1000 http and 500 https proxies)

import asyncio
import re
import sys
import pprint
import json

from proxybroker import Broker

def __get_proxies(protocol, limit):
	proxy_list = []

	async def save(proxies):
	    while True:
	        proxy = await proxies.get()
	        if proxy is None: 
	        	break
	        match = re.search('([\d]+\.[\d]+\.[\d]+\.[\d]+:[\d]+)', str(proxy))
	        proxy_list.append(match.group(1))
	        print('Count:', len(proxy_list))

	proxies = asyncio.Queue()
	broker = Broker(proxies)
	tasks = asyncio.gather(broker.find(types=[protocol.upper()], limit=limit), save(proxies))
	loop = asyncio.get_event_loop().run_until_complete(tasks)

	return proxy_list

if __name__ == "__main__":
	if len(sys.argv) < 3:
		sys.exit('Not enough arguments')
		
	http_proxies = __get_proxies('http', int(sys.argv[1]))
	https_proxies = __get_proxies('https', int(sys.argv[2]))

	proxies = {}
	try:
		with open('resources/proxies.txt', 'r') as f:
			proxies = json.load(f)
	except:
		proxies['http'] = []
		proxies['https'] = []

	with open('resources/proxies.txt', 'w') as f:
		proxies['http'] = list(set(proxies['http'] + http_proxies))
		proxies['https'] = list(set(proxies['https'] + https_proxies))

		print(json.dumps(proxies), file=f)

	
	
