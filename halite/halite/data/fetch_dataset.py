# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import time
import json
from src.api.KaggleClient import KaggleClient


@click.command()
@click.argument('output_filepath', type=click.Path(), default='data/raw/')
def main(output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    fetch_dataset(output_filepath)


def fetch_dataset(output_filepath):

    EPISODE_WATERMARK = 1100
    TEAM_WATERMARK = 25
    REQUEST_LIMIT = 10  # 60*3 # must be smaller than 1000
    REQUEST_DISCOVERY_BUDGET = 0.5
    ARBITRARY_TEAM_ID = '5118174'

    DOWNLOAD_FILEPATH = Path.cwd().joinpath(output_filepath)
    DOWNLOAD_FILEPATH.mkdir(parents=True, exist_ok=True)
    METADATA_DIR = Path.cwd().joinpath('scraper_metadata')
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    EPISODE_PMAP_FILEPATH = str(
        METADATA_DIR.joinpath('episode_priority_map.txt'))
    TEAM_PMAP_FILEPATH = str(METADATA_DIR.joinpath('team_priority_map.txt'))
    EPISODE_DOWNLOAD_FILEPATH = str(
        METADATA_DIR.joinpath('episode_downloads.txt'))

    request_budget = REQUEST_LIMIT
    teams_priorities = synchronize_disk(TEAM_PMAP_FILEPATH, dict())
    episodes_priorities = synchronize_disk(EPISODE_PMAP_FILEPATH, dict())
    scraped_episodes = synchronize_disk(EPISODE_DOWNLOAD_FILEPATH, set())
    scraped_teams = set()

    teams_queue = get_min_weighted_priority_queue(
        teams_priorities, TEAM_WATERMARK)
    episodes_queue = get_max_weighted_priority_queue(
        teams_priorities, EPISODE_WATERMARK, request_budget)

    api = KaggleClient()
    request_counter = 0
    request_start = time.time()

    if 0 < len(teams_queue):
        logging.warn('Discovering episodes by prioritized team queue')
    else:
        logging.info(
            'Discovering episodes by arbitrary team_id {}'.format(
                ARBITRARY_TEAM_ID))
        teams_queue = [(ARBITRARY_TEAM_ID, -1)]

    logging.info('Scraping started.')
    is_scraping = True
    while is_scraping:
        # scrape episodes
        while 0 < len(episodes_queue) and 0 < request_budget:
            episode_id, episode_priority = episodes_queue.pop(0)
            logging.info(
                'Requesting replay for Episode ID {} with priority {}'.format(
                    episode_id, episode_priority))

            # TODO: os.environ['DOWNLOAD_FILEPATH']
            filepath = str(DOWNLOAD_FILEPATH.joinpath(
                'replay_EPISODEID_{}_{}.json'.format(episode_id, time.time())))
            response = api.replay.episode(episode_id)
            save_json(filepath, response)

            logging.info('Downloaded Episode ID {}'.format(episode_id))
            scraped_episodes.add(episode_id)
            episodes_priorities.pop(episode_id, None)

            request_budget = request_budget - 1
            request_counter = request_counter + 1
            if 60 <= request_counter:
                idle_time = 60 - (time.time() - request_start)
                if 0 < idle_time:
                    logging.info('Idling for {} seconds'.format(idle_time))
                    time.sleep(idle_time)
                request_start = time.time()
                request_counter = 0

        # find episodes
        discovery_limit = (request_budget * REQUEST_DISCOVERY_BUDGET)//1
        discovery_budget = discovery_limit
        while 0 < discovery_budget and 0 < len(teams_queue):
            team_id, team_priority = teams_queue.pop(0)
            logging.info(
                'Requesting metadata for Team ID {} with priority {}'.format(
                    team_id, team_priority))

            # TODO: os.environ['DOWNLOAD_FILEPATH']
            filepath = str(DOWNLOAD_FILEPATH.joinpath(
                'metadata_TEAMID_{}_{}.json'.format(team_id, time.time())))
            response = api.episodes.team(team_id)
            save_json(filepath, response)
            logging.info('Downloaded metadata by Team ID {}'.format(team_id))
            scraped_teams.add(team_id)

            teams_priorities.update(get_team_priority_map(response))
            # TODO: os.environ
            teams_priorities = synchronize_priority_map(
                teams_priorities, TEAM_PMAP_FILEPATH, scraped_teams)
            # TODO: os.environ
            teams_queue = get_min_weighted_priority_queue(
                teams_priorities, TEAM_WATERMARK)

            episodes_priorities.update(get_episode_priority_map(response))
            # TODO: os.environ
            episodes_priorities = synchronize_priority_map(
                episodes_priorities, EPISODE_PMAP_FILEPATH, scraped_episodes)
            # TODO: os.environ
            episodes_queue = get_max_weighted_priority_queue(
                episodes_priorities, EPISODE_WATERMARK, request_budget)
            discovery_budget = discovery_budget - 1

            request_counter = request_counter + 1
            if 60 <= request_counter:
                idle_time = 60 - (time.time() - request_start)
                if 0 < idle_time:
                    logging.info('Idling for {} seconds'.format(idle_time))
                    time.sleep(idle_time)
                request_start = time.time()
                request_counter = 0

        request_budget = request_budget - discovery_limit + discovery_budget

        # update scrape state
        is_scraping = 0 < len(episodes_queue) and 0 < request_budget

    synchronize_disk(TEAM_PMAP_FILEPATH, teams_priorities, overwrite=True)
    synchronize_disk(EPISODE_PMAP_FILEPATH,
                     episodes_priorities, overwrite=True)
    synchronize_disk(EPISODE_DOWNLOAD_FILEPATH,
                     scraped_episodes, overwrite=True)
    logging.info('Scraping finished')


def synchronize_list_to_disk(filepath, data, overwrite):
    if not overwrite:
        try:
            with open(filepath, 'r') as f:
                data.update([int(x) for x in f.readlines()])
        except FileNotFoundError:
            logging.warn(
                'Failed to synchronize from disk. File {} is empty.'.format(
                    filepath))
    try:
        with open(filepath, 'w+') as f:
            for x in data:
                f.write('{}\n'.format(x))
    except Exception:
        logging.exception('Failed to synchronize to disk.')
    return data


def synchronize_dict_to_disk(filepath, data, overwrite):
    if not overwrite:
        try:
            with open(filepath, 'r') as f:
                data.update(json.load(f))
        except FileNotFoundError:
            logging.warn(
                'Failed to synchronize from disk. File {} is empty.'.format(
                    filepath))
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f)
    except Exception:
        logging.exception('Failed to synchronize to disk.')
    return data


def synchronize_disk(filepath, data, overwrite=False):
    if isinstance(data, list) or isinstance(data, set):
        synchronized_data = synchronize_list_to_disk(filepath, data, overwrite)
    elif isinstance(data, dict):
        synchronized_data = synchronize_dict_to_disk(filepath, data, overwrite)
    return synchronized_data


def get_team_ids(episode_metadata):
    team_ids = {episode_metadata['result']['teams'][i]['id']
                for i in range(len(episode_metadata['result']['teams']))}
    team_ids.update({agent['submission']['teamId']
                    for episode in episode_metadata['result']['episodes']
                    for agent in episode['agents']})
    return team_ids


def get_episode_ids(episode_metadata):
    episode_ids = {episode['id']
                   for episode in episode_metadata['result']['episodes']}
    return episode_ids


def save_json(filepath, json_data):
    with open(filepath, 'w') as f:
        json.dump(json_data, f)
        # f.write(json_data)


def get_team_priority_map(episode_metadata):
    priority_map = dict()
    for team in episode_metadata['result']['teams']:
        team_id = str(team['id'])
        priority = team['publicLeaderboardRank']
        priority_map[team_id] = priority
    return priority_map


def get_episode_priority_map(episode_metadata):
    priority_map = dict()
    for episode in episode_metadata['result']['episodes']:
        episode_id = str(episode['id'])
        weights = [agent['updatedScore']
                   for agent in episode['agents']
                   if None is not agent['updatedScore']]
        priority = sum(weights)/len(weights) if 0 < len(weights) else 0
        priority_map[episode_id] = priority
    return priority_map


def synchronize_priority_map(priority_map, filepath, stale_priorities=None):
    if None is stale_priorities:
        stale_priorities = set()
    priority_map = synchronize_disk(filepath, priority_map)
    priority_map = {k: v for k, v in priority_map.items()
                    if k not in stale_priorities}
    priority_map = synchronize_disk(filepath, priority_map, overwrite=True)
    return priority_map


def get_min_weighted_priority_queue(
    priority_map,
    watermark,
    request_budget=None
):
    return get_priority_queue(
        priority_map,
        watermark,
        max_weighted=False,
        request_budget=request_budget
    )


def get_max_weighted_priority_queue(
    priority_map,
    watermark,
    request_budget=None
):
    return get_priority_queue(
        priority_map,
        watermark,
        max_weighted=True,
        request_budget=request_budget
    )


def get_priority_queue(
    priorty_map,
    watermark,
    max_weighted=True,
    request_budget=None
):
    prioritized_keys = sorted(
        priorty_map, key=priorty_map.get, reverse=max_weighted)

    def watermark_fn(x): return (max_weighted and watermark <=
                                 x) or (not max_weighted and watermark >= x)
    watermarked_keys = [
        k for k in prioritized_keys if watermark_fn(priorty_map[k])]
    ratelimited_keys = watermarked_keys[:request_budget]
    return [(k, priorty_map[k]) for k in ratelimited_keys]


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
