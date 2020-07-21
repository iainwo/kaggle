from urllib.parse import urljoin
import inspect
import requests
import json as json_lib


class KaggleClient(object):
    """ A Kaggle Halite Episode Service API wrapper. """

    def __init__(self):
        self.requester = self.Requester()
        self._attach_endpoints()

    def _attach_endpoints(self):
        """ Generate and attach endpoints """
        for name, endpoint in inspect.getmembers(self):
            if (inspect.isclass(endpoint)
                and issubclass(endpoint, self._Endpoint)
                    and endpoint is not self._Endpoint):
                endpoint_instance = endpoint(self.requester)
                setattr(self, endpoint.name, endpoint_instance)

    class Requester(object):
        """ An object for making API requests """

        def GET(self, url, params=None, json=None):
            if None is params:
                params = dict()
            if None is json:
                json = dict()

            params.setdefault('datatype', 'json')
            response = requests.post(url, params=params, json=json)
            if 200 != response.status_code:
                error = 'HTTPError: {}'.format(response.status_code)
                return {'success': False, 'error': error}
            try:
                return response.json()
            except ValueError as err:
                return {'success': False, 'error': err}

    class _Endpoint(object):
        """ Base class of an endpoint """
        url = 'https://www.kaggle.com/requests/EpisodeService/'

        def __init__(self, requester):
            self.requester = requester

        def _GET(self, path, params=None, json=None):
            request_url = urljoin(self.url, path)
            return self.requester.GET(request_url, params, json)

    class Replay(_Endpoint):
        name = 'replay'

        def episode(self, episode_id, params=None):
            json = {"EpisodeId": episode_id}
            resp = self._GET('GetEpisodeReplay', params, json)
            resp['result']['replay'] = json_lib.loads(resp['result']['replay'])
            return resp

    class Episodes(_Endpoint):
        name = 'episodes'

        def episodes(self, episode_ids, params=None):
            json = {'Ids': episode_ids}
            return self._GET('ListEpisodes', params, json)

        def team(self, team_id, params=None):
            json = {'TeamId': team_id}
            return self._GET('ListEpisodes', params, json)

        def submission(self, submission_id, params=None):
            json = {'SubmissionId': submission_id}
            return self._GET('ListEpisodes', params, json)
