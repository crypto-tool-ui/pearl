FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04

RUN useradd --create-home --uid 10001 pearl
COPY target/release/pearl-miner /usr/local/bin/pearl-miner
COPY config.example.toml /etc/pearl-miner/config.toml
RUN chown -R pearl:pearl /etc/pearl-miner
USER pearl
WORKDIR /etc/pearl-miner
ENTRYPOINT ["/usr/local/bin/pearl-miner"]
