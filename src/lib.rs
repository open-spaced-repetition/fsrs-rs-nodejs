#![deny(clippy::all)]
#![allow(unexpected_cfgs)]
use napi::JsNumber;
use napi::bindgen_prelude::{AsyncTask, JsFunction, Result};
use std::sync::{Arc, Mutex};

mod train_task;
use train_task::{ComputeParametersTask, ProgressData};

// https://github.com/rust-lang/rust-analyzer/issues/17429
use napi_derive::napi;

#[napi(js_name = "FSRS")]
#[derive(Debug)]
pub struct FSRS(Arc<Mutex<fsrs::FSRS>>);
#[napi]
pub const FSRS5_DEFAULT_DECAY: f32 = 0.5;
#[napi]
pub const FSRS6_DEFAULT_DECAY: f32 = 0.1542;
/// FSRS-7 default parameters (34 values), the same as `fsrs::DEFAULT_PARAMETERS`.
///
/// These are the parameters used by `new FSRS()` and `new FSRS([])`.
#[napi]
pub const DEFAULT_PARAMETERS: [f32; 34] = fsrs::DEFAULT_PARAMETERS;
/// Alias of {@link DEFAULT_PARAMETERS}: the FSRS-7 default parameters (34 values).
#[napi]
pub const FSRS7_DEFAULT_PARAMETERS: [f32; 34] = DEFAULT_PARAMETERS;
/// FSRS-6 default parameters (21 values), the same as `fsrs::FSRS6_DEFAULT_PARAMETERS`.
#[napi]
pub const FSRS6_DEFAULT_PARAMETERS: [f32; 21] = fsrs::FSRS6_DEFAULT_PARAMETERS;

/// The FSRS algorithm version used by a model.
#[napi(string_enum)]
pub enum ModelVersion {
  #[napi(value = "FSRS-6")]
  Fsrs6,
  #[napi(value = "FSRS-7")]
  Fsrs7,
}

impl From<fsrs::ModelVersion> for ModelVersion {
  fn from(version: fsrs::ModelVersion) -> Self {
    match version {
      fsrs::ModelVersion::Fsrs6 => Self::Fsrs6,
      fsrs::ModelVersion::Fsrs7 => Self::Fsrs7,
    }
  }
}

impl From<ModelVersion> for fsrs::ComputeParametersVersion {
  fn from(version: ModelVersion) -> Self {
    match version {
      ModelVersion::Fsrs6 => Self::Fsrs6,
      ModelVersion::Fsrs7 => Self::Fsrs7,
    }
  }
}

fn js_numbers_to_f32(values: Vec<JsNumber>) -> Result<Vec<f32>> {
  values
    .iter()
    .map(|value| value.get_double().map(|value| value as f32))
    .collect()
}

fn js_number_to_usize(value: &JsNumber) -> Option<usize> {
  value
    .get_int64()
    .ok()
    .and_then(|value| usize::try_from(value).ok())
}

fn js_number_to_u64(value: &JsNumber) -> Option<u64> {
  value
    .get_int64()
    .ok()
    .and_then(|value| u64::try_from(value).ok())
}

fn js_number_to_f64(value: &JsNumber) -> Option<f64> {
  value.get_double().ok().filter(|value| value.is_finite())
}

fn napi_error(message: impl Into<String>) -> napi::Error {
  napi::Error::from_reason(message.into())
}

fn fsrs_error(action: &str, error: fsrs::FSRSError) -> napi::Error {
  napi_error(format!("FSRS {action} failed: {error}"))
}

fn validate_training_config(config: fsrs::TrainingConfig) -> Result<fsrs::TrainingConfig> {
  if config.batch_size == 0 {
    return Err(napi_error("batchSize must be greater than 0"));
  }
  if !config.learning_rate.is_finite() {
    return Err(napi_error("learningRate must be finite"));
  }
  if !config.gamma.is_finite() {
    return Err(napi_error("gamma must be finite"));
  }
  Ok(config)
}

fn vec_to_f32(values: Vec<JsNumber>) -> Result<Vec<f32>> {
  js_numbers_to_f32(values)
}

fn vec_to_array<const N: usize>(values: Vec<f64>, name: &str) -> Result<[f32; N]> {
  if values.len() != N {
    return Err(napi_error(format!(
      "{name} must contain exactly {N} values"
    )));
  }

  let mut array = [0.0; N];
  for (index, value) in values.into_iter().enumerate() {
    if !value.is_finite() {
      return Err(napi_error(format!("{name} must contain finite values")));
    }
    array[index] = value as f32;
  }

  Ok(array)
}

fn matrix_to_array<const ROWS: usize, const COLS: usize>(
  values: Vec<Vec<f64>>,
  name: &str,
) -> Result<[[f32; COLS]; ROWS]> {
  if values.len() != ROWS {
    return Err(napi_error(format!(
      "{name} must contain exactly {ROWS} rows"
    )));
  }

  let mut matrix = [[0.0; COLS]; ROWS];
  for (row_index, row) in values.into_iter().enumerate() {
    matrix[row_index] = vec_to_array(row, name)?;
  }

  Ok(matrix)
}

fn array_to_vec<const N: usize>(values: [f32; N]) -> Vec<f64> {
  values.into_iter().map(f64::from).collect()
}

fn matrix_to_vec<const ROWS: usize, const COLS: usize>(
  values: [[f32; COLS]; ROWS],
) -> Vec<Vec<f64>> {
  values.into_iter().map(array_to_vec).collect()
}

/// FSRS-7 has no parameter pre-training, so training without short-term memory would freeze the
/// initial stability at its defaults. FSRS-7 therefore always enables it and ignores `false`.
fn enable_short_term_from_options(
  options: Option<&ComputeParametersOption>,
  model_version: fsrs::ComputeParametersVersion,
) -> bool {
  model_version == fsrs::ComputeParametersVersion::Fsrs7
    || options.and_then(|x| x.enable_short_term).unwrap_or(true)
}

fn num_relearning_steps_from_options(options: Option<&ComputeParametersOption>) -> Option<usize> {
  options
    .and_then(|x| x.num_relearning_steps.as_ref())
    .and_then(js_number_to_usize)
}

fn progress_timeout_from_options(options: Option<&ComputeParametersOption>) -> u64 {
  options
    .and_then(|x| x.timeout.as_ref())
    .and_then(js_number_to_u64)
    .unwrap_or(500)
}

fn card_ids_from_options(options: Option<&ComputeParametersOption>) -> Result<Option<Vec<i64>>> {
  options
    .and_then(|x| x.card_ids.as_ref())
    .map(|card_ids| {
      card_ids
        .iter()
        .map(|card_id| card_id.get_int64())
        .collect::<Result<Vec<_>>>()
    })
    .transpose()
}

/// Defaults to FSRS-7, like `fsrs::ComputeParametersVersion::default()`.
fn model_version_from_options(
  options: Option<&ComputeParametersOption>,
) -> fsrs::ComputeParametersVersion {
  options
    .and_then(|x| x.model_version)
    .map_or_else(fsrs::ComputeParametersVersion::default, Into::into)
}

/// Schedule penalties only exist in FSRS-7, so they are always disabled for FSRS-6.
fn enable_sched_penalties_from_options(
  options: Option<&ComputeParametersOption>,
  model_version: fsrs::ComputeParametersVersion,
) -> bool {
  model_version == fsrs::ComputeParametersVersion::Fsrs7
    && options
      .and_then(|x| x.enable_sched_penalties)
      .unwrap_or(false)
}

fn training_config_from_options(
  options: Option<&ComputeParametersOption>,
  model_version: fsrs::ComputeParametersVersion,
) -> Result<Option<fsrs::TrainingConfig>> {
  let Some(options) = options.and_then(|x| x.training_config.as_ref()) else {
    // Without a training config, fsrs 7 trains FSRS-6 with the FSRS-7 hyperparameters.
    // Pass the default config to keep the FSRS-6 defaults (5 epochs, learning rate 0.04).
    return Ok(
      (model_version == fsrs::ComputeParametersVersion::Fsrs6).then(fsrs::TrainingConfig::default),
    );
  };
  let mut config = fsrs::TrainingConfig::default();

  if let Some(value) = options.num_epochs.as_ref().and_then(js_number_to_usize) {
    config.num_epochs = value;
  }
  if let Some(value) = options.batch_size.as_ref().and_then(js_number_to_usize) {
    config.batch_size = value;
  }
  if let Some(value) = options.seed.as_ref().and_then(js_number_to_u64) {
    config.seed = value;
  }
  if let Some(value) = options.learning_rate.as_ref().and_then(js_number_to_f64) {
    config.learning_rate = value;
  }
  if let Some(value) = options.max_seq_len.as_ref().and_then(js_number_to_usize) {
    config.max_seq_len = value;
  }
  if let Some(value) = options.gamma.as_ref().and_then(js_number_to_f64) {
    config.gamma = value;
  }

  Ok(Some(validate_training_config(config)?))
}

fn compute_parameters_input(
  train_set: Vec<&FSRSItem>,
  options: Option<&ComputeParametersOption>,
  progress: Option<Arc<Mutex<fsrs::CombinedProgressState>>>,
) -> Result<fsrs::ComputeParametersInput> {
  let model_version = model_version_from_options(options);
  Ok(fsrs::ComputeParametersInput {
    train_set: train_set.into_iter().map(|x| x.0.clone()).collect(),
    card_ids: card_ids_from_options(options)?,
    progress,
    enable_short_term: enable_short_term_from_options(options, model_version),
    enable_sched_penalties: enable_sched_penalties_from_options(options, model_version),
    model_version,
    num_relearning_steps: num_relearning_steps_from_options(options),
    training_config: training_config_from_options(options, model_version)?,
  })
}

fn convert_starting_states(
  starting_states: Option<Vec<Option<&MemoryState>>>,
  len: usize,
) -> Vec<Option<fsrs::MemoryState>> {
  starting_states.map_or_else(
    || (0..len).map(|_| None).collect(),
    |states| {
      states
        .into_iter()
        .map(|state| state.map(|state| state.0))
        .collect()
    },
  )
}

#[napi]
impl FSRS {
  /// - Parameters must be provided before running commands that need them.
  /// - Parameters may be omitted or an empty array to use the FSRS-7 defaults
  ///   ({@link DEFAULT_PARAMETERS}).
  /// - The parameter count selects the model version: 17, 19 or 21 values select FSRS-6
  ///   (17 and 19 are expanded to 21), 34 values select FSRS-7.
  #[napi(constructor, catch_unwind)]
  pub fn new(parameters: Option<Vec<JsNumber>>) -> Result<Self> {
    let params = js_numbers_to_f32(parameters.unwrap_or_default())?;
    let model = fsrs::FSRS::new(&params)
      .map_err(|e| napi::Error::from_reason(format!("FSRS initialization failed: {e}")))?;
    Ok(Self(Arc::new(Mutex::new(model))))
  }

  /// The model version selected by the parameters passed to the constructor.
  #[napi(getter)]
  pub fn version(&self) -> ModelVersion {
    self.0.lock().unwrap().version().into()
  }

  /// The intervals and memory states for each answer button.
  ///
  /// `daysElapsed` may be fractional. FSRS-7 uses fractional days, while FSRS-6 rounds them to
  /// whole days.
  ///
  /// Parameters must have been provided when calling [`new FSRS()`]{@link constructor}.
  #[napi]
  pub fn next_states(
    &self,
    current_memory_state: Option<&MemoryState>,
    desired_retention: f64,
    days_elapsed: f64,
  ) -> Result<NextStates> {
    let locked_model = self.0.lock().unwrap();
    // The FSRS-6 model rounds the elapsed days itself.
    Ok(NextStates(
      locked_model
        .next_states_with_elapsed_days(
          current_memory_state.map(|x| x.0),
          desired_retention as f32,
          days_elapsed as f32,
        )
        .map_err(|e| fsrs_error("nextStates", e))?,
    ))
  }

  /// For FSRS-7, the fast stability is assumed to equal `stability`.
  #[napi]
  pub fn next_interval(&self, stability: Option<f64>, desired_retention: f64, rating: u32) -> f32 {
    let locked_model = self.0.lock().unwrap();
    locked_model.next_interval(
      stability.map(|value| value as f32),
      desired_retention as f32,
      rating,
    )
  }

  /// Determine how well the model and parameters predict performance.
  ///
  /// Parameters must have been provided when calling [`new FSRS()`]{@link constructor}.
  #[napi]
  pub fn evaluate(&self, train_set: Vec<&FSRSItem>) -> Result<ModelEvaluation> {
    // Convert your `JS` training items to owned `fsrs::FSRSItem`
    let train_data = train_set
      .into_iter()
      .map(|item| item.0.clone())
      .collect::<Vec<_>>();

    let locked_model = self.0.lock().unwrap();
    let result = locked_model
      .evaluate(train_data, |_| true)
      .map_err(|e| napi::Error::from_reason(format!("FSRS evaluate failed: {e}")))?;
    Ok(result.into())
  }

  /// If a card has incomplete learning history, memory state can be approximated from
  /// current sm2 values.
  ///
  /// Parameters must have been provided when calling [`new FSRS()`]{@link constructor}.
  #[napi]
  pub fn memory_state_from_sm2(
    &self,
    ease_factor: f64,
    interval: f64,
    sm2_retention: f64,
  ) -> Result<MemoryState> {
    let locked_model = self.0.lock().unwrap();
    Ok(MemoryState(
      locked_model
        .memory_state_from_sm2(ease_factor as f32, interval as f32, sm2_retention as f32)
        .map_err(|e| fsrs_error("memoryStateFromSm2", e))?,
    ))
  }

  /// Calculate the current memory state for a given card's history of reviews.
  /// In the case of truncated reviews, `startingState` can be set to the value of
  /// {@link memoryStateFromSm2} for the first review (which should not be included
  /// in {@link FSRSItem}). If not provided, the card starts as new.
  ///
  /// Parameters must have been provided when calling [`new FSRS()`]{@link constructor}.
  #[napi]
  pub fn memory_state(
    &self,
    item: &FSRSItem,
    starting_state: Option<&MemoryState>,
  ) -> Result<MemoryState> {
    let locked_model = self.0.lock().unwrap();
    Ok(MemoryState(
      locked_model
        .memory_state(item.0.clone(), starting_state.map(|x| x.0))
        .map_err(|e| fsrs_error("memoryState", e))?,
    ))
  }

  #[napi]
  pub fn memory_state_batch(
    &self,
    items: Vec<&FSRSItem>,
    #[napi(ts_arg_type = "Array<MemoryState | null | undefined>")] starting_states: Option<
      Vec<Option<&MemoryState>>,
    >,
  ) -> Result<Vec<MemoryState>> {
    let starting_states = convert_starting_states(starting_states, items.len());
    let locked_model = self.0.lock().unwrap();
    locked_model
      .memory_state_batch(
        items.into_iter().map(|item| item.0.clone()).collect(),
        starting_states,
      )
      .map(|states| states.into_iter().map(MemoryState).collect())
      .map_err(|e| fsrs_error("memoryStateBatch", e))
  }

  #[napi]
  pub fn historical_memory_states(
    &self,
    item: &FSRSItem,
    starting_state: Option<&MemoryState>,
  ) -> Result<Vec<MemoryState>> {
    let locked_model = self.0.lock().unwrap();
    locked_model
      .historical_memory_states(item.0.clone(), starting_state.map(|x| x.0))
      .map(|states| states.into_iter().map(MemoryState).collect())
      .map_err(|e| fsrs_error("historicalMemoryStates", e))
  }

  #[napi]
  pub fn historical_memory_state_batch(
    &self,
    items: Vec<&FSRSItem>,
    #[napi(ts_arg_type = "Array<MemoryState | null | undefined>")] starting_states: Option<
      Vec<Option<&MemoryState>>,
    >,
  ) -> Result<Vec<Vec<MemoryState>>> {
    let locked_model = self.0.lock().unwrap();
    locked_model
      .historical_memory_state_batch(
        items.into_iter().map(|item| item.0.clone()).collect(),
        starting_states.map(|states| {
          states
            .into_iter()
            .map(|state| state.map(|state| state.0))
            .collect()
        }),
      )
      .map(|rows| {
        rows
          .into_iter()
          .map(|row| row.into_iter().map(MemoryState).collect())
          .collect()
      })
      .map_err(|e| fsrs_error("historicalMemoryStateBatch", e))
  }
}

#[napi(js_name = "FSRSReview")]
#[derive(Debug)]
pub struct FSRSReview(fsrs::FSRSReview);

#[napi]
impl FSRSReview {
  /// `deltaT` may be fractional. FSRS-7 uses fractional days, while FSRS-6 rounds them to
  /// whole days when updating memory states and when training.
  #[napi(constructor)]
  pub fn new(rating: u32, delta_t: f64) -> Self {
    Self(fsrs::FSRSReview {
      rating,
      delta_t: delta_t as f32,
    })
  }
  /// 1-4
  #[napi(getter)]
  pub fn rating(&self) -> u32 {
    self.0.rating
  }
  /// The number of days that passed (can be fractional)
  /// # Warning
  /// `delta_t` for item first(initial) review must be 0
  #[napi(getter)]
  pub fn delta_t(&self) -> f64 {
    self.0.delta_t as f64
  }
  #[napi(js_name = "toJSON")]
  pub fn to_json(&self) -> String {
    format!("{:?}", self.0)
  }
}

/// Stores a list of reviews for a card, in chronological order. Each FSRSItem corresponds
/// to a single review, but contains the previous reviews of the card as well, after the
/// first one.
///
/// When used during review, the last item should include the correct `delta_t`, but
/// the provided rating is ignored as all four ratings are returned by `.nextStates()`
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

  #[napi(setter)]
  pub fn set_reviews(&mut self, reviews: Vec<&FSRSReview>) {
    self.0.reviews = reviews.iter().map(|x| x.0).collect();
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
  /// `stabilityFast` is only used by FSRS-7 and defaults to `stability`.
  #[napi(constructor)]
  pub fn new(stability: f64, difficulty: f64, stability_fast: Option<f64>) -> Self {
    Self(fsrs::MemoryState {
      stability: stability as f32,
      difficulty: difficulty as f32,
      stability_fast: stability_fast.unwrap_or(stability) as f32,
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
  /// The FSRS-7 fast stability. FSRS-6 ignores it and returns states where it equals `stability`.
  #[napi(getter)]
  pub fn stability_fast(&self) -> f64 {
    self.0.stability_fast as f64
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

/// Omitted fields use the defaults of `fsrs::TrainingConfig`.
#[napi(object)]
pub struct TrainingConfig {
  /// 0 (the default) means the model version's default: 9 for FSRS-7, 5 for FSRS-6.
  pub num_epochs: Option<JsNumber>,
  pub batch_size: Option<JsNumber>,
  pub seed: Option<JsNumber>,
  /// 0 (the default) means the model version's default: 0.0118 for FSRS-7, 0.04 for FSRS-6.
  pub learning_rate: Option<JsNumber>,
  pub max_seq_len: Option<JsNumber>,
  pub gamma: Option<JsNumber>,
}

#[napi(object)]
pub struct SimulationResult {
  pub memorized_cnt_per_day: Vec<f64>,
  pub review_cnt_per_day: Vec<u32>,
  pub learn_cnt_per_day: Vec<u32>,
  pub cost_per_day: Vec<f64>,
  pub correct_cnt_per_day: Vec<u32>,
  /// The desired retention passed to `simulate`, or `undefined` when no card was learned or
  /// reviewed. Every simulated card uses the same desired retention.
  pub average_desired_retention: Option<f64>,
  pub introduced_cnt_per_day: Vec<u32>,
}

impl SimulationResult {
  /// fsrs 7 no longer reports the average desired retention. `simulate` gives every card the
  /// same desired retention, so the average is that value whenever any card was simulated.
  fn new(result: fsrs::SimulationResult, desired_retention: f32) -> Self {
    let simulated_cards = result.learn_cnt_per_day.iter().sum::<usize>()
      + result.review_cnt_per_day.iter().sum::<usize>();
    Self {
      memorized_cnt_per_day: result
        .memorized_cnt_per_day
        .into_iter()
        .map(f64::from)
        .collect(),
      review_cnt_per_day: result
        .review_cnt_per_day
        .iter()
        .map(|&value| value as u32)
        .collect(),
      learn_cnt_per_day: result
        .learn_cnt_per_day
        .iter()
        .map(|&value| value as u32)
        .collect(),
      cost_per_day: result.cost_per_day.into_iter().map(f64::from).collect(),
      correct_cnt_per_day: result
        .correct_cnt_per_day
        .iter()
        .map(|&value| value as u32)
        .collect(),
      average_desired_retention: (simulated_cards > 0).then_some(f64::from(desired_retention)),
      introduced_cnt_per_day: result
        .introduced_cnt_per_day
        .iter()
        .map(|&value| value as u32)
        .collect(),
    }
  }
}

#[napi(object)]
pub struct SimulatorConfig {
  pub deck_size: u32,
  pub learn_span: u32,
  pub max_cost_perday: f64,
  pub max_ivl: f64,
  pub first_rating_prob: Vec<f64>,
  pub review_rating_prob: Vec<f64>,
  pub learn_limit: u32,
  pub review_limit: u32,
  pub new_cards_ignore_review_limit: bool,
  pub learning_step_transitions: Vec<Vec<f64>>,
  pub relearning_step_transitions: Vec<Vec<f64>>,
  pub state_rating_costs: Vec<Vec<f64>>,
  pub learning_step_count: u32,
  pub relearning_step_count: u32,
  pub suspend_after_lapses: Option<u32>,
}

impl SimulatorConfig {
  fn into_fsrs(self) -> Result<fsrs::SimulatorConfig> {
    Ok(fsrs::SimulatorConfig {
      deck_size: self.deck_size as usize,
      learn_span: self.learn_span as usize,
      max_cost_perday: self.max_cost_perday as f32,
      max_ivl: self.max_ivl as f32,
      first_rating_prob: vec_to_array(self.first_rating_prob, "firstRatingProb")?,
      review_rating_prob: vec_to_array(self.review_rating_prob, "reviewRatingProb")?,
      learn_limit: self.learn_limit as usize,
      review_limit: self.review_limit as usize,
      new_cards_ignore_review_limit: self.new_cards_ignore_review_limit,
      suspend_after_lapses: self.suspend_after_lapses,
      post_scheduling_fn: None,
      review_priority_fn: None,
      review_rating_cost_fn: None,
      learning_step_transitions: matrix_to_array(
        self.learning_step_transitions,
        "learningStepTransitions",
      )?,
      relearning_step_transitions: matrix_to_array(
        self.relearning_step_transitions,
        "relearningStepTransitions",
      )?,
      state_rating_costs: matrix_to_array(self.state_rating_costs, "stateRatingCosts")?,
      learning_step_count: self.learning_step_count as usize,
      relearning_step_count: self.relearning_step_count as usize,
    })
  }
}

impl From<fsrs::SimulatorConfig> for SimulatorConfig {
  fn from(config: fsrs::SimulatorConfig) -> Self {
    Self {
      deck_size: config.deck_size as u32,
      learn_span: config.learn_span as u32,
      max_cost_perday: config.max_cost_perday as f64,
      max_ivl: config.max_ivl as f64,
      first_rating_prob: array_to_vec(config.first_rating_prob),
      review_rating_prob: array_to_vec(config.review_rating_prob),
      learn_limit: config.learn_limit as u32,
      review_limit: config.review_limit as u32,
      new_cards_ignore_review_limit: config.new_cards_ignore_review_limit,
      learning_step_transitions: matrix_to_vec(config.learning_step_transitions),
      relearning_step_transitions: matrix_to_vec(config.relearning_step_transitions),
      state_rating_costs: matrix_to_vec(config.state_rating_costs),
      learning_step_count: config.learning_step_count as u32,
      relearning_step_count: config.relearning_step_count as u32,
      suspend_after_lapses: config.suspend_after_lapses,
    }
  }
}

#[napi(object)]
pub struct ModelEvaluation {
  pub log_loss: f64,
  pub rmse_bins: f64,
}

impl From<fsrs::ModelEvaluation> for ModelEvaluation {
  fn from(result: fsrs::ModelEvaluation) -> Self {
    Self {
      log_loss: result.log_loss as f64,
      rmse_bins: result.rmse_bins as f64,
    }
  }
}

#[napi(object)]
pub struct ComputeParametersOption {
  /// Model version to train or evaluate. Defaults to FSRS-7, matching fsrs-rs.
  #[napi(ts_type = "`${ModelVersion}`")]
  pub model_version: Option<ModelVersion>,
  /// Whether to enable short-term memory parameters. Defaults to true. Only FSRS-6 can disable
  /// them: FSRS-7 always enables short-term memory and ignores `false`.
  pub enable_short_term: Option<bool>,
  /// Whether to enable the FSRS-7 schedule penalties during training. Defaults to false;
  /// ignored by FSRS-6.
  pub enable_sched_penalties: Option<bool>,
  /// Number of relearning steps
  pub num_relearning_steps: Option<JsNumber>,
  /// Optional card ids aligned with `trainSet`.
  #[napi(ts_type = "Array<number>")]
  pub card_ids: Option<Vec<JsNumber>>,
  /// Optional optimizer hyperparameters
  pub training_config: Option<TrainingConfig>,
  #[napi(
    ts_type = "(err: Error | null , value: { current: number, total: number, percent: number }) => void"
  )]
  pub progress: Option<JsFunction>,
  #[napi(ts_type = "number")]
  pub timeout: Option<JsNumber>,
}

#[napi(js_name = "FilterOutlierResult")]
pub struct FilterOutlierResult {
  dataset_for_initialization: Vec<fsrs::FSRSItem>,
  trainset: Vec<fsrs::FSRSItem>,
}

#[napi]
impl FilterOutlierResult {
  #[napi(getter)]
  pub fn dataset_for_initialization(&self) -> Vec<FSRSItem> {
    self
      .dataset_for_initialization
      .iter()
      .cloned()
      .map(FSRSItem)
      .collect()
  }

  #[napi(getter)]
  pub fn trainset(&self) -> Vec<FSRSItem> {
    self.trainset.iter().cloned().map(FSRSItem).collect()
  }
}

#[napi]
pub fn default_simulator_config() -> SimulatorConfig {
  fsrs::SimulatorConfig::default().into()
}

#[napi]
pub fn simulate(
  w: Vec<JsNumber>,
  desired_retention: f64,
  config: Option<SimulatorConfig>,
  seed: Option<JsNumber>,
) -> Result<SimulationResult> {
  let config = match config {
    Some(config) => config.into_fsrs()?,
    None => fsrs::SimulatorConfig::default(),
  };
  let seed = seed
    .as_ref()
    .map(|seed| {
      js_number_to_u64(seed).ok_or_else(|| napi_error("seed must be a non-negative integer"))
    })
    .transpose()?;

  let desired_retention = desired_retention as f32;
  fsrs::simulate(&config, &vec_to_f32(w)?, desired_retention, seed, None)
    .map(|result| SimulationResult::new(result, desired_retention))
    .map_err(|e| fsrs_error("simulate", e))
}

/// Calculate appropriate parameters for the provided review history.
///
/// `options.modelVersion` selects the version to train and defaults to FSRS-7.
#[napi(ts_return_type = "Promise<Array<number>>")]
pub fn compute_parameters(
  train_set: Vec<&FSRSItem>,
  #[napi(ts_arg_type = "ComputeParametersOption")] options: Option<ComputeParametersOption>,
) -> Result<AsyncTask<ComputeParametersTask>> {
  // Convert your `JS` training items to owned `fsrs::FSRSItem`
  let train_data = train_set
    .into_iter()
    .map(|item| item.0.clone())
    .collect::<Vec<_>>();

  // Turn `JsFunction` into a `ThreadsafeFunction`
  let fn_form_js = if let Some(callback) = options.as_ref().and_then(|x| x.progress.as_ref()) {
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

  let model_version = model_version_from_options(options.as_ref());
  let task = ComputeParametersTask {
    train_data,
    card_ids: card_ids_from_options(options.as_ref())?,
    enable_short_term: enable_short_term_from_options(options.as_ref(), model_version),
    enable_sched_penalties: enable_sched_penalties_from_options(options.as_ref(), model_version),
    model_version,
    num_relearning_steps: num_relearning_steps_from_options(options.as_ref()),
    training_config: training_config_from_options(options.as_ref(), model_version)?,
    progress_callback: fn_form_js,
    progress_timeout: progress_timeout_from_options(options.as_ref()),
  };

  Ok(AsyncTask::new(task))
}

#[napi]
pub fn benchmark(
  train_set: Vec<&FSRSItem>,
  #[napi(ts_arg_type = "ComputeParametersOption")] options: Option<ComputeParametersOption>,
) -> Result<Vec<f32>> {
  Ok(fsrs::benchmark(compute_parameters_input(
    train_set,
    options.as_ref(),
    None,
  )?))
}

#[napi]
pub fn evaluate_with_time_series_splits(
  train_set: Vec<&FSRSItem>,
  #[napi(ts_arg_type = "ComputeParametersOption")] options: Option<ComputeParametersOption>,
) -> Result<ModelEvaluation> {
  let result = fsrs::evaluate_with_time_series_splits(
    compute_parameters_input(train_set, options.as_ref(), None)?,
    |_| true,
  )
  .map_err(|e| fsrs_error("evaluateWithTimeSeriesSplits", e))?;

  Ok(result.into())
}

#[napi]
pub fn filter_outlier(
  dataset_for_initialization: Vec<&FSRSItem>,
  trainset: Vec<&FSRSItem>,
) -> Result<FilterOutlierResult> {
  if dataset_for_initialization
    .iter()
    .chain(trainset.iter())
    .any(|item| item.0.reviews.is_empty())
  {
    return Err(napi_error("FSRSItem reviews must not be empty"));
  }

  let (dataset_for_initialization, trainset) = fsrs::filter_outlier(
    dataset_for_initialization
      .into_iter()
      .map(|item| item.0.clone())
      .collect(),
    trainset.into_iter().map(|item| item.0.clone()).collect(),
  );

  Ok(FilterOutlierResult {
    dataset_for_initialization,
    trainset,
  })
}

#[napi]
pub fn check_and_fill_parameters(parameters: Vec<JsNumber>) -> Result<Vec<f32>> {
  fsrs::check_and_fill_parameters(&vec_to_f32(parameters)?)
    .map_err(|e| fsrs_error("checkAndFillParameters", e))
}
