#![deny(clippy::all)]
use napi::bindgen_prelude::{AsyncTask, JsFunction, Result};
use napi::{Env, JsNumber};
use std::sync::{Arc, Mutex};

mod train_task;
use train_task::{ComputeParametersTask, ProgressData};

// https://github.com/rust-lang/rust-analyzer/issues/17429
use napi_derive::napi;

#[napi(js_name = "FSRS")]
#[derive(Debug)]
pub struct FSRS(Arc<Mutex<fsrs::FSRS>>);

#[napi]
/// directly use fsrs::DEFAULT_PARAMETERS will cause error.
/// referencing statics in constants is unstable
/// see issue #119618 <https://github.com/rust-lang/rust/issues/119618> for more information
/// `static` and `const` variables can refer to other `const` variables. A `const` variable, however, cannot refer to a `static` variable.
/// to fix this, the value can be extracted to a `const` and then used.
pub const DEFAULT_PARAMETERS: [f32; 19] = [
  0.40255, 1.18385, 3.173, 15.69105, 7.1949, 0.5345, 1.4604, 0.0046, 1.54575, 0.1192, 1.01925,
  1.9395, 0.11, 0.29605, 2.2698, 0.2315, 2.9898, 0.51655, 0.6621,
];
impl Default for FSRS {
  fn default() -> Self {
    Self::new(None)
  }
}

#[napi]
impl FSRS {
  #[napi(constructor)]
  pub fn new(parameters: Option<Vec<JsNumber>>) -> Self {
    let params: [f32; 19] = match parameters {
      Some(parameters) => {
        let mut array = [0.0; 19];
        for (i, value) in parameters.iter().enumerate().take(19) {
          array[i] = value.get_double().unwrap_or(0.0) as f32;
        }
        array
      }
      None => DEFAULT_PARAMETERS,
    };
    Self(Arc::new(Mutex::new(
      fsrs::FSRS::new(Some(&params)).unwrap(),
    )))
  }

  #[napi(ts_return_type = "Promise<Array<number>>")]
  pub fn compute_parameters(
    &self,
    train_set: Vec<&FSRSItem>,
    enable_short_term: bool,
    #[napi(
      ts_arg_type = "(err: null | Error, value: { current: number, total: number, percent: number }) => void"
    )]
    progress_js_fn: Option<JsFunction>,
    #[napi(ts_arg_type = "number")] timeout: Option<JsNumber>,
  ) -> Result<AsyncTask<ComputeParametersTask>> {
    // Convert your `JS` training items to owned `fsrs::FSRSItem`
    let train_data = train_set
      .into_iter()
      .map(|item| item.0.clone())
      .collect::<Vec<_>>();

    // Turn `JsFunction` into a `ThreadsafeFunction`
    let fn_form_js = if let Some(callback) = progress_js_fn {
      Some(callback.create_threadsafe_function(0, |ctx| {
        let progress_data: ProgressData = ctx.value;
        let env = ctx.env;
        let current = env.create_uint32(progress_data.current as u32)?;
        let total = env.create_uint32(progress_data.total as u32)?;
        let percent = env.create_double(progress_data.percent)?;
        let mut progress_obj = env.create_object()?;
        progress_obj.set_named_property("current", current)?;
        progress_obj.set_named_property("total", total)?;
        progress_obj.set_named_property("percent", percent)?;
        Ok(vec![progress_obj])
      })?)
    } else {
      None
    };

    let task = ComputeParametersTask {
      model: Arc::clone(&self.0),
      train_data,
      enable_short_term,
      progress_callback: fn_form_js,
      progress_timeout: timeout
        .map(|x| x.get_int64().unwrap_or(500) as u64)
        .unwrap_or(500),
    };

    Ok(AsyncTask::new(task))
  }

  #[napi]
  pub fn next_states(
    &self,
    current_memory_state: Option<&MemoryState>,
    desired_retention: f64,
    days_elapsed: u32,
  ) -> NextStates {
    let locked_model = self.0.lock().unwrap();
    NextStates(
      locked_model
        .next_states(
          current_memory_state.map(|x| x.0),
          desired_retention as f32,
          days_elapsed,
        )
        .unwrap(),
    )
  }

  #[napi]
  pub fn benchmark(&self, train_set: Vec<&FSRSItem>) -> Vec<f32> {
    let locked_model = self.0.lock().unwrap();
    locked_model.benchmark(train_set.iter().map(|x| x.0.clone()).collect(), true)
  }

  #[napi]
  pub fn evaluate(&self, env: Env, train_set: Vec<&FSRSItem>) -> Result<ModelEvaluation> {
    // Convert your `JS` training items to owned `fsrs::FSRSItem`
    let train_data = train_set
      .into_iter()
      .map(|item| item.0.clone())
      .collect::<Vec<_>>();

    let locked_model = self.0.lock().unwrap();
    let result = locked_model
      .evaluate(train_data, |_| true)
      .map_err(|e| napi::Error::from_reason(format!("FSRS evaluate failed: {}", e)))?;
    Ok(ModelEvaluation {
      log_loss: env.create_double(result.log_loss as f64)?,
      rmse_bins: env.create_double(result.rmse_bins as f64)?,
    })
  }

  #[napi]
  pub fn memory_state_from_sm2(
    &self,
    ease_factor: f64,
    interval: f64,
    sm2_retention: f64,
  ) -> MemoryState {
    let locked_model = self.0.lock().unwrap();
    MemoryState(
      locked_model
        .memory_state_from_sm2(ease_factor as f32, interval as f32, sm2_retention as f32)
        .unwrap(),
    )
  }

  #[napi]
  pub fn memory_state(&self, item: &FSRSItem, starting_state: Option<&MemoryState>) -> MemoryState {
    let locked_model = self.0.lock().unwrap();
    MemoryState(
      locked_model
        .memory_state(item.0.clone(), starting_state.map(|x| x.0))
        .unwrap(),
    )
  }
}

#[napi(js_name = "FSRSReview")]
#[derive(Debug)]
pub struct FSRSReview(fsrs::FSRSReview);

#[napi]
impl FSRSReview {
  #[napi(constructor)]
  pub fn new(rating: u32, delta_t: u32) -> Self {
    Self(fsrs::FSRSReview { rating, delta_t })
  }
  #[napi(getter)]
  pub fn rating(&self) -> u32 {
    self.0.rating
  }
  #[napi(getter)]
  pub fn delta_t(&self) -> u32 {
    self.0.delta_t
  }
  #[napi(js_name = "toJSON")]
  pub fn to_json(&self) -> String {
    format!("{:?}", self.0)
  }
}

#[napi(js_name = "FSRSItem")]
#[derive(Debug)]
pub struct FSRSItem(fsrs::FSRSItem);
#[napi]
impl FSRSItem {
  #[napi(constructor)]
  pub fn new(reviews: Vec<&FSRSReview>) -> Self {
    Self(fsrs::FSRSItem {
      reviews: reviews.iter().map(|x| x.0).collect(),
    })
  }

  #[napi(getter)]
  pub fn reviews(&self) -> Vec<FSRSReview> {
    self.0.reviews.iter().map(|x| FSRSReview(*x)).collect()
  }

  #[napi]
  pub fn long_term_review_cnt(&self) -> u32 {
    self.0.long_term_review_cnt() as u32
  }

  #[napi(js_name = "toJSON")]
  pub fn to_json(&self) -> String {
    format!("{:?}", self.0)
  }
}

#[napi(js_name = "MemoryState")]
#[derive(Debug)]
pub struct MemoryState(fsrs::MemoryState);
#[napi]
impl MemoryState {
  #[napi(constructor)]
  pub fn new(stability: f64, difficulty: f64) -> Self {
    Self(fsrs::MemoryState {
      stability: stability as f32,
      difficulty: difficulty as f32,
    })
  }
  #[napi(getter)]
  pub fn stability(&self) -> f64 {
    self.0.stability as f64
  }
  #[napi(getter)]
  pub fn difficulty(&self) -> f64 {
    self.0.difficulty as f64
  }
  #[napi(js_name = "toJSON")]
  pub fn to_json(&self) -> String {
    format!("{:?}", self.0)
  }
}

#[napi(js_name = "NextStates")]
#[derive(Debug)]
pub struct NextStates(fsrs::NextStates);
#[napi]
impl NextStates {
  #[napi(getter)]
  pub fn hard(&self) -> ItemState {
    ItemState(self.0.hard.clone())
  }
  #[napi(getter)]
  pub fn good(&self) -> ItemState {
    ItemState(self.0.good.clone())
  }
  #[napi(getter)]
  pub fn easy(&self) -> ItemState {
    ItemState(self.0.easy.clone())
  }
  #[napi(getter)]
  pub fn again(&self) -> ItemState {
    ItemState(self.0.again.clone())
  }
  #[napi(js_name = "toJSON")]
  pub fn to_json(&self) -> String {
    format!("{:?}", self.0)
  }
}

#[napi(js_name = "ItemState")]
#[derive(Debug)]
pub struct ItemState(fsrs::ItemState);
#[napi]
impl ItemState {
  #[napi(getter)]
  pub fn memory(&self) -> MemoryState {
    MemoryState(self.0.memory)
  }
  #[napi(getter)]
  pub fn interval(&self) -> f32 {
    self.0.interval
  }
  #[napi(js_name = "toJSON")]
  pub fn to_json(&self) -> String {
    format!("{:?}", self.0)
  }
}

#[napi(object)]
pub struct ModelEvaluation {
  pub log_loss: JsNumber,
  pub rmse_bins: JsNumber,
}
