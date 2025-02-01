use napi::bindgen_prelude::{Env, Error, Result, Status, Task};
use napi::threadsafe_function::{ErrorStrategy, ThreadsafeFunction, ThreadsafeFunctionCallMode};
use std::sync::{Arc, Mutex};

#[derive(Debug)]
pub struct ProgressData {
  pub current: usize,
  pub total: usize,
  pub percent: f64,
}

/// A background task that runs `compute_parameters`, sending progress updates via TSFN.
pub struct ComputeParametersTask {
  // Thread-safe reference to your FSRS model
  pub(crate) model: Arc<Mutex<fsrs::FSRS>>,
  // Training data, made owned so it doesn't reference `&self`
  pub(crate) enable_short_term: bool,
  pub(crate) train_data: Vec<fsrs::FSRSItem>,
  // The threadsafe JS callback for partial updates
  pub(crate) progress_callback:
    Option<ThreadsafeFunction<ProgressData, ErrorStrategy::CalleeHandled>>,

  pub(crate) progress_timeout: u64,
}

impl Task for ComputeParametersTask {
  type Output = Vec<f32>;
  type JsValue = Vec<f64>;

  fn compute(&mut self) -> Result<Self::Output> {
    // 1) Create a shared progress object
    let progress_state = fsrs::CombinedProgressState::new_shared();
    let progress_state_for_thread = Arc::clone(&progress_state);
    // Clone what we need for the separate thread
    let train_data = self.train_data.clone();
    let model = Arc::clone(&self.model);
    let enable_short_term = self.enable_short_term;

    // 2) Spawn a new thread that does the heavy lifting
    //    so we can poll progress *in parallel* on this thread.
    let handle = std::thread::spawn(move || -> Result<Vec<f32>> {
      let locked = model.lock().map_err(|_| {
        Error::new(
          Status::GenericFailure,
          "Failed to lock FSRS model".to_string(),
        )
      })?;

      // Now use `progress_state_for_thread` inside the closure
      locked
        .compute_parameters(
          train_data,
          Some(progress_state_for_thread),
          enable_short_term,
        )
        .map_err(|e| Error::new(Status::GenericFailure, format!("{:?}", e)))
    });

    // 3) Meanwhile, on *this* thread, poll `progress_state` in a loop
    //    and call `progress_callback` with updated progress.
    if self.progress_callback.is_some() {
      loop {
        let (current, total, finished) = {
          let p = progress_state.lock().unwrap();
          (p.current(), p.total(), p.finished())
        };

        let percent = if total == 0 {
          0.0
        } else {
          (current as f64 / total as f64) as f64
        };

        // Call JS callback if you want once per second or whenever progress changes
        let status = self.progress_callback.as_ref().unwrap().call(
          Ok(ProgressData {
            current,
            total,
            percent,
          }),
          ThreadsafeFunctionCallMode::NonBlocking,
        );

        if status != napi::Status::Ok {
          eprintln!("Failed to call JS callback, status = {:?}", status);
        }

        if finished || percent >= 100.0 {
          break;
        }

        // Sleep briefly before polling again
        std::thread::sleep(std::time::Duration::from_millis(self.progress_timeout));
      }
    }

    // 4) Join the compute thread to get the final result
    let final_result = handle.join().map_err(|_| {
      Error::new(
        Status::GenericFailure,
        "Panic occurred in compute thread".to_string(),
      )
    })??; // `??` to unwrap the `Result` from inside the thread

    // 5) Return the final result
    Ok(final_result)
  }

  fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
    Ok(output.iter().map(|&x| x as f64).collect())
  }
}
