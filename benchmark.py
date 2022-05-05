import requests
import pandas as pd

from concurrent.futures import ThreadPoolExecutor, wait


DOCKER_TEST_HOST = "0.0.0.0"
DOCKER_TEST_PORT = 5000
PODMAN_TEST_HOST = DOCKER_TEST_HOST
PODMAN_TEST_PORT = 5001
TEST_IMAGE_PATH = "data/test.jpg"
TEST_IMAGE = open(TEST_IMAGE_PATH, "rb")
N_MAX_REQUESTS = 1000


def docker_predict() -> requests.Response:
    res = requests.post(f"http://{DOCKER_TEST_HOST}:{DOCKER_TEST_PORT}/predict", files=dict(image=open(TEST_IMAGE_PATH, "rb")))
    return res


def podman_predict() -> requests.Response:
    res = requests.post(f"http://{PODMAN_TEST_HOST}:{PODMAN_TEST_PORT}/predict", files=dict(image=open(TEST_IMAGE_PATH, "rb")))
    return res


def test_docker_single_request():
    res = docker_predict()
    return res.elapsed.microseconds / 100


def test_podman_single_request():
    res = podman_predict()
    return res.elapsed.microseconds / 100


def test_seq_requests():
    docker_ms = list()
    podman_ms = list()

    for _ in range(N_MAX_REQUESTS):
        docker_ms.append(test_docker_single_request())
        podman_ms.append(test_podman_single_request())

    pd.DataFrame([docker_ms, podman_ms], columns=["docker_elapsed_ms", "podman_elapsed_ms"]).to_csv("data/results/seq.csv", index_label="request")


def test_sim_requests():
    docker_ms = list()
    podman_ms = list()

    with ThreadPoolExecutor(max_workers=N_MAX_REQUESTS) as p:
        # Docker
        futures = [p.submit(test_docker_single_request) for _ in range(N_MAX_REQUESTS)]
        wait(futures)
        docker_ms = [x.result() for x in futures]
        # Podman
        futures = [p.submit(test_podman_single_request) for _ in range(N_MAX_REQUESTS)]
        wait(futures)
        docker_ms = [x.result() for x in futures]

    pd.DataFrame([docker_ms, podman_ms], columns=["docker_elapsed_ms", "podman_elapsed_ms"]).to_csv("data/results/sim.csv", index_label="request")


test_seq_requests()
test_sim_requests()
