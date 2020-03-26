use crate::ffi;

/// QuEST Environment
pub struct QuestEnv {
    env: ffi::QuESTEnv,
}

impl QuestEnv {
    pub fn new() -> Self {
        unsafe {
            QuestEnv {
                env: ffi::createQuESTEnv(),
            }
        }
    }

    pub fn sync(&mut self) {
        unsafe {
            ffi::syncQuESTEnv(self.env);
        }
    }

    pub fn report(&self) {
        unsafe {
            ffi::reportQuESTEnv(self.env);
        }
    }
}

impl Drop for QuestEnv {
    fn drop(&mut self) {
        unsafe {
            ffi::destroyQuESTEnv(self.env);
        }
    }
}

impl From<QuestEnv> for ffi::QuESTEnv {
    fn from(item: QuestEnv) -> Self {
        item.env
    }
}

impl From<&QuestEnv> for ffi::QuESTEnv {
    fn from(item: &QuestEnv) -> Self {
        item.env
    }
}

pub fn seed_quest(seed_values: Vec<u64>) {
    unsafe {
        ffi::seedQuEST(seed_values.as_ptr() as *mut u64, seed_values.len() as i32);
    }
}

/// Seed the Mersenne Twister used for random number generation in the QuEST
/// environment with an example defualt seed.
/// 
/// This default seeding function uses the mt19937 init_by_array function with
/// two keys -- time and pid. Subsequent calls to mt19937 genrand functions will
/// use this seeding. For a multi process code, the same seed is given to all
/// process, therefore this seeding is only appropriate to use for functions such
/// as measure where all processes require the same random value.
///
/// For more information about the MT, see:
/// http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/MT2002/emt19937ar.html.
pub fn seed_quest_default() {
    unsafe {
        ffi::seedQuESTDefault();
    }
}

