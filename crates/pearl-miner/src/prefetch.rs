//! Demand-driven, one-attempt lookahead. Dropping either side unblocks shutdown.
use anyhow::{Context, Result};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::thread::{self, JoinHandle};

pub struct Prefetch<I, O> {
    requests: Option<SyncSender<I>>,
    results: Option<Receiver<O>>,
    worker: Option<JoinHandle<()>>,
}

impl<I: Send + 'static, O: Send + 'static> Prefetch<I, O> {
    pub fn new(mut prepare: impl FnMut(I) -> O + Send + 'static) -> Self {
        let (requests, input) = mpsc::sync_channel(1);
        let (output, results) = mpsc::sync_channel(0);
        let worker = thread::spawn(move || {
            while let Ok(request) = input.recv() {
                if output.send(prepare(request)).is_err() {
                    break;
                }
            }
        });
        Self {
            requests: Some(requests),
            results: Some(results),
            worker: Some(worker),
        }
    }

    pub fn request(&self, input: I) -> Result<()> {
        self.requests
            .as_ref()
            .unwrap()
            .send(input)
            .map_err(|_| anyhow::anyhow!("matrix preparation worker disconnected"))
    }

    pub fn receive(&self) -> Result<O> {
        self.results
            .as_ref()
            .unwrap()
            .recv()
            .context("matrix preparation worker failed")
    }
}

impl<I, O> Drop for Prefetch<I, O> {
    fn drop(&mut self) {
        // Drop the receiver before joining: the worker may be waiting to deliver a result.
        self.results.take();
        self.requests.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prepares_next_attempt_while_consumer_is_busy() {
        let (done, ready) = mpsc::channel();
        let pipeline = Prefetch::new(move |attempt| {
            done.send(attempt).unwrap();
            attempt + 1
        });
        pipeline.request(0).unwrap();
        assert_eq!(pipeline.receive().unwrap(), 1);
        assert_eq!(ready.recv().unwrap(), 0);
        pipeline.request(1).unwrap();
        // Completion must not depend on the consumer asking for the next result.
        assert_eq!(
            ready
                .recv_timeout(std::time::Duration::from_secs(5))
                .unwrap(),
            1
        );
        assert_eq!(pipeline.receive().unwrap(), 2);
    }
    #[test]
    fn shutdown_discards_unconsumed_attempt() {
        let (done, ready) = mpsc::channel();
        let pipeline = Prefetch::new(move |()| {
            done.send(()).unwrap();
            42
        });
        pipeline.request(()).unwrap();
        ready.recv().unwrap();
        drop(pipeline);
    }
    #[test]
    fn preparation_errors_reach_consumer() {
        let pipeline = Prefetch::new(|()| -> Result<()> { anyhow::bail!("bad header") });
        pipeline.request(()).unwrap();
        assert!(pipeline
            .receive()
            .unwrap()
            .unwrap_err()
            .to_string()
            .contains("bad header"));
    }
}
