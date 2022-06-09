import requests
import pandas as pd
import argparse
import socket
import os
import datetime

from concurrent.futures import ThreadPoolExecutor, wait
from typing import List

TEST_IMAGE_PATH = "data/image.jpg"
TEST_IMAGE = open(TEST_IMAGE_PATH, "rb")
MAX_REQUESTS = 1000


def post(ip: str, port: int=5000) -> requests.Response:
    return requests.post(
        f"http://{ip}:{port}/", files=dict(image=open(TEST_IMAGE_PATH, "rb"))
    )


def benchmark_request(ip: str, port: int=5000) -> float:
    res = post(ip, port)
    return res.elapsed.microseconds / 1000


def benchmark_sequential_requests(ips: List[str], port: int=5000) -> pd.DataFrame:
    def _sequential_loop(ip: str, port: int=5000) -> pd.Series:
        ns = list()

        for _ in range(MAX_REQUESTS):
            ns.append(benchmark_request(ip, port))

        return pd.Series(ns, name=ip)


    df = pd.DataFrame()

    with ThreadPoolExecutor() as p:
        futures = list()

        for ip in ips:
            futures.append(p.submit(_sequential_loop, ip=ip, port=port))

        wait(futures)
        results = [x.result() for x in futures]

        for i, r in enumerate(results):
            if i == 0:
                df = r.to_frame()
            else:
                df = df.join(r)

    return df


def benchmark_simultaneous_requests(ips: List[str], port: int=5000) -> pd.DataFrame:
    def _simultaneous_loop(ip: str, port: int=5000) -> pd.Series:
        ns = list()

        with ThreadPoolExecutor(max_workers=MAX_REQUESTS // 10) as p:
            futures = list()

            for _ in range(MAX_REQUESTS):
                futures.append(p.submit(benchmark_request, ip=ip, port=port))

            wait(futures)
            ns = [x.result() for x in futures]

        return pd.Series(ns, name=ip)


    df = pd.DataFrame()

    with ThreadPoolExecutor() as p:
        futures = list()

        for ip in ips:
            futures.append(p.submit(_simultaneous_loop, ip=ip, port=port))

        wait(futures)
        results = [x.result() for x in futures]

        for i, r in enumerate(results):
            if i == 0:
                df = r.to_frame()
            else:
                df = df.join(r)

    return df


if __name__ == "__main__":
    def _get_current_datetime() -> str:
        cdt = datetime.datetime.now()
        return f"{cdt.year}{cdt.month}{cdt.day}{cdt.hour}{cdt.minute}{cdt.second}"


    parser = argparse.ArgumentParser(description="Benchmark APIs")
    parser.add_argument("-i", metavar="192.168.1.2", type=str, required=True, help="IP addresses to benchmark")
    parser.add_argument("-p", metavar=5000, type=int, default=5000, help="Port to use for benchmarking")
    parser.add_argument("-n", metavar=1000, type=int, default=100, help="Maximum number of requests to perform during a test")
    parser.add_argument("-o", metavar="results", type=str, default="results", help="Directory to output results to")

    args = parser.parse_args()
    ips = args.i.split()
    port = args.p
    MAX_REQUESTS = args.n
    out = args.o

    if not os.path.isdir(out):
        os.mkdir(out)

    # Print Benchmark info
    print(f"Benchmarking the following IPs at port {port}:")
    
    for ip in ips:
        try:
            socket.inet_aton(ip)
            print("  -", ip)
        except:
            print("Illegal IP:", ip)
            print("Exiting...")
            exit(1)

    # Test sequentially
    print("Testing sequentially....")
    seq = benchmark_sequential_requests(ips, port)
    seq.to_csv(os.path.join(out, f"{_get_current_datetime()}_{MAX_REQUESTS}_seq.csv"), index_label="request")

    # Test simultaneously
    print("Testing simultaneously....")
    sim = benchmark_simultaneous_requests(ips, port)
    sim.to_csv(os.path.join(out, f"{_get_current_datetime()}_{MAX_REQUESTS}_sim.csv"), index_label="request")
