"""HTTP and on-disk session checks using only hypothetical fixture teams."""

import copy
import http.client
import json
import tempfile
import threading
import unittest
from pathlib import Path

from src.prediction import demo_artifact
from src.simulator_app import LocalHTTPServer, TournamentService, handler_for
from src.tournament import new_session


class AppTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name)/'session.json'
        self.artifact = demo_artifact()
        self.service = TournamentService(self.artifact,self.path)
        handler = handler_for(self.service)
        handler.log_message = lambda *args: None
        self.server = LocalHTTPServer(('127.0.0.1',0),handler)
        self.thread = threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.directory.cleanup()

    def request(self,method,path,payload=None):
        connection=http.client.HTTPConnection('127.0.0.1',self.server.server_port,timeout=10)
        connection.request(method,path,body=json.dumps(payload) if payload is not None else None,
                           headers={'Content-Type':'application/json'} if payload is not None else {})
        response=connection.getresponse()
        body=response.read().decode()
        status=response.status
        connection.close()
        return status,body

    def test_browser_routes_result_undo_save_and_resume(self):
        status,html=self.request('GET','/')
        self.assertEqual(status,200)
        self.assertIn('Title chances',html)
        self.assertIn('No official participants',html)
        teams=list(self.artifact['state']['global_ratings'])
        bracket={'label':'Hypothetical HTTP test','quarterfinals':[teams[i:i+2] for i in range(0,8,2)]}
        status,body=self.request('POST','/api/bracket',{'bracket':bracket})
        self.assertEqual(status,200,body)
        self.assertTrue(self.path.exists())
        event={'series':'QF1','winner':teams[0],'timestamp':'2020-01-01T00:00:00Z','event_id':'synthetic-http-game'}
        status,body=self.request('POST','/api/game',event)
        self.assertEqual(status,200,body)
        self.assertEqual(json.loads(body)['series'][0]['score'],[1,0])
        status,body=self.request('POST','/api/game',event)
        self.assertEqual(status,400)
        self.assertEqual(len(self.service.session['events']),1)
        status,body=self.request('POST','/api/bracket',{'bracket':bracket})
        self.assertEqual(status,400)
        resumed=TournamentService(demo_artifact(),self.path)
        self.assertEqual(resumed.status()['series'][0]['score'],[1,0])
        status,body=self.request('POST','/api/simulate',{'simulations':1000,'seed':1})
        self.assertEqual(status,200,body)
        self.assertAlmostEqual(sum(row['title_probability'] for row in json.loads(body)['teams']),1)
        status,download=self.request('GET','/api/session')
        self.assertEqual(status,200)
        status,body=self.request('POST','/api/undo',{})
        self.assertEqual(status,200,body)
        self.assertEqual(json.loads(body)['series'][0]['score'],[0,0])
        self.assertEqual(TournamentService(self.artifact,self.path).status()['session']['events'],[])
        status,body=self.request('POST','/api/import',{'session':json.loads(download)})
        self.assertEqual(status,200,body)
        self.assertEqual(json.loads(body)['series'][0]['score'],[1,0])

    def test_invalid_import_does_not_replace_state(self):
        teams=list(self.artifact['state']['global_ratings'])
        session=new_session(self.artifact,{'label':'Hypothetical test','quarterfinals':[teams[i:i+2] for i in range(0,8,2)]})
        self.service.save(session)
        before=self.path.read_bytes()
        changed=copy.deepcopy(session)
        changed['artifact_sha256']='wrong'
        status,body=self.request('POST','/api/import',{'session':changed})
        self.assertEqual(status,400,body)
        self.assertEqual(self.path.read_bytes(),before)
        self.assertEqual(self.service.session,session)

    def test_another_server_cannot_claim_the_same_port(self):
        with self.assertRaises(OSError):
            LocalHTTPServer(('127.0.0.1', self.server.server_port), handler_for(self.service))


if __name__=='__main__':
    unittest.main()
