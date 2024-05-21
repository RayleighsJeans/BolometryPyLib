def getProgId(T0):
    import json
    import urllib.request as urllib

    host = 'archive-webapi.ipp-hgw.mpg.de'
    path = \
        'ArchiveDB/raw/W7X/ProjectDesc.1/ProgramLabelStart/' + \
        'parms/ProgramLogLabel/progId'
    url = 'http://%s/%s/_signal.json?from=%d&upto=%d'%(host,path,T0-11e9,T0)
    return json.load(urllib.urlopen(url,timeout=1))['values'][0]

def ecrh_duration(YYYYMMDD,timeout=10,default=10):
     import urllib.request as urllib
     import json,time

     fp = \
         urllib.urlopen(
             "http://sv-coda-wsvc-3.ipp-hgw.mpg.de/last_trigger")
     mybytes = fp.read()
     mystr = mybytes.decode("utf8")
     fp.close()

     T0 = int(mystr)
     progid = getProgId(T0)
     url = \
        'https://w7x-logbook.ipp-hgw.mpg.de/api/log/XP_%d.%d'%(
            int(YYYYMMDD),progid)
     TIC = time.time()
     while True:
        tic = time.time()
        try:
            res = urllib.urlopen(url)
        except urllib.HTTPError:
            toc = time.time()
            if toc-TIC>timeout:
                return default
            time.sleep(max(0,toc-tic+.1))
            continue
        else:
            break
     d = json.loads(res.read())
     tags = d['_source']['tags']
     for i in range(len(tags)):
         if tags[i]['name'] == 'ECRH duration':
             return float(tags[i]['value'])
     return 1
  