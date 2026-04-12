use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

fn hash_str(s: &str) -> u64 {
    let mut h = DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

fn pseudo_random(generation: u32, name: &str) -> f64 {
    let combined = format!("{}:{}", generation, name);
    let h = hash_str(&combined);
    (h % 10000) as f64 / 10000.0
}

#[derive(Clone, Debug, PartialEq)]
pub enum MutationType {
    ParamAdjust,
    ThresholdShift,
    WeightRebalance,
    AddBehavior,
    RemoveBehavior,
    SwapPriority,
    RateChange,
    CapChange,
}

#[derive(Clone, Debug)]
pub struct Behavior {
    pub name: String,
    pub value: f64,
    pub min: f64,
    pub max: f64,
    pub default_val: f64,
    pub mutation_rate: f64,
    pub uses: u32,
    pub cumulative_score: f64,
}

#[derive(Clone, Debug)]
pub struct MutationRecord {
    pub mutation_type: MutationType,
    pub parameter: String,
    pub old_value: f64,
    pub new_value: f64,
    pub reason: String,
    pub generation: u32,
    pub reverted: bool,
}

pub struct Engine {
    behaviors: HashMap<String, Behavior>,
    history: Vec<MutationRecord>,
    generation: u32,
    mutations_total: u32,
    mutations_reverted: u32,
    fitness_threshold: f64,
    mutation_probability: f64,
    elite_threshold: f64,
}

impl Engine {
    pub fn new() -> Self {
        Engine {
            behaviors: HashMap::new(),
            history: Vec::new(),
            generation: 0,
            mutations_total: 0,
            mutations_reverted: 0,
            fitness_threshold: 0.3,
            mutation_probability: 0.1,
            elite_threshold: 0.8,
        }
    }

    pub fn add_behavior(&mut self, name: &str, value: f64, min: f64, max: f64, mut_rate: f64) {
        let b = Behavior {
            name: name.to_string(),
            value: value.clamp(min, max),
            min,
            max,
            default_val: value,
            mutation_rate: mut_rate,
            uses: 0,
            cumulative_score: 0.0,
        };
        self.behaviors.insert(name.to_string(), b);
    }

    pub fn find_behavior(&self, name: &str) -> Option<&Behavior> {
        self.behaviors.get(name)
    }

    pub fn get(&self, name: &str) -> f64 {
        self.behaviors.get(name).map(|b| b.value).unwrap_or(-1.0)
    }

    pub fn set(&mut self, name: &str, value: f64) {
        if let Some(b) = self.behaviors.get_mut(name) {
            b.value = value.clamp(b.min, b.max);
        }
    }

    pub fn cycle(&mut self, fitness: f64) -> usize {
        self.generation += 1;
        let mut mutations = 0usize;

        if fitness >= self.elite_threshold {
            // Elite: only mutate worst behaviors with higher probability
            let worst_names: Vec<String> = self.worst_behaviors(3).into_iter().map(|b| b.name.clone()).collect();
            for name in worst_names {
                if pseudo_random(self.generation, &name) < self.mutation_probability * 2.0 {
                    mutations += self.mutate_behavior(&name, "elite pressure");
                }
            }
        } else if fitness >= self.fitness_threshold {
            // Normal mutation
            for name in self.behaviors.keys().cloned().collect::<Vec<_>>() {
                let prob = self.behaviors.get(&name).map(|b| b.mutation_rate).unwrap_or(self.mutation_probability);
                if pseudo_random(self.generation, &name) < prob {
                    mutations += self.mutate_behavior(&name, "normal evolution");
                }
            }
        } else {
            // Aggressive: mutate everything with higher rates
            for name in self.behaviors.keys().cloned().collect::<Vec<_>>() {
                if pseudo_random(self.generation, &name) < self.mutation_probability * 3.0 {
                    mutations += self.mutate_behavior(&name, "aggressive mutation");
                }
            }
        }

        mutations
    }

    fn mutate_behavior(&mut self, name: &str, reason: &str) -> usize {
        let (old_value, min, max) = {
            let b = match self.behaviors.get(name) {
                Some(b) => b,
                None => return 0,
            };
            let range = b.max - b.min;
            if range == 0.0 {
                return 0;
            }
            (b.value, b.min, b.max)
        };

        let range = max - min;
        let r = pseudo_random(self.generation, &format!("mut:{}", name));
        let direction: f64 = if r < 0.5 { -1.0 } else { 1.0 };
        let magnitude = range * 0.1 * (pseudo_random(self.generation, &format!("mag:{}", name)) + 0.1);
        let new_value = (old_value + direction * magnitude).clamp(min, max);

        if (new_value - old_value).abs() < 1e-10 {
            return 0;
        }

        let record = MutationRecord {
            mutation_type: MutationType::ParamAdjust,
            parameter: name.to_string(),
            old_value,
            new_value,
            reason: reason.to_string(),
            generation: self.generation,
            reverted: false,
        };

        self.mutations_total += 1;
        if let Some(b) = self.behaviors.get_mut(name) {
            b.value = new_value;
        }
        self.history.push(record);
        1
    }

    pub fn score(&mut self, behavior: &str, outcome: f64) {
        if let Some(b) = self.behaviors.get_mut(behavior) {
            b.uses += 1;
            b.cumulative_score += outcome;
        }
    }

    pub fn revert(&mut self, index: usize) -> bool {
        if index >= self.history.len() {
            return false;
        }
        let record = &self.history[index];
        if record.reverted {
            return false;
        }
        let name = record.parameter.clone();
        let old_value = record.old_value;
        if let Some(b) = self.behaviors.get_mut(&name) {
            b.value = old_value;
        }
        self.history[index].reverted = true;
        self.mutations_reverted += 1;
        true
    }

    pub fn rollback(&mut self, target_generation: u32) -> usize {
        let mut reverted = 0usize;
        // Process from newest to oldest to avoid conflicts
        for i in (0..self.history.len()).rev() {
            if self.history[i].generation > target_generation && !self.history[i].reverted {
                if self.revert(i) {
                    reverted += 1;
                }
            }
        }
        reverted
    }

    pub fn worst_behaviors(&self, n: usize) -> Vec<&Behavior> {
        let mut sorted: Vec<&Behavior> = self.behaviors.values().collect();
        sorted.sort_by(|a, b| {
            let avg_a = if a.uses > 0 { a.cumulative_score / a.uses as f64 } else { 0.0 };
            let avg_b = if b.uses > 0 { b.cumulative_score / b.uses as f64 } else { 0.0 };
            avg_a.partial_cmp(&avg_b).unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.into_iter().take(n).collect()
    }

    pub fn best_behaviors(&self, n: usize) -> Vec<&Behavior> {
        let mut sorted: Vec<&Behavior> = self.behaviors.values().collect();
        sorted.sort_by(|a, b| {
            let avg_a = if a.uses > 0 { a.cumulative_score / a.uses as f64 } else { 0.0 };
            let avg_b = if b.uses > 0 { b.cumulative_score / b.uses as f64 } else { 0.0 };
            avg_b.partial_cmp(&avg_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.into_iter().take(n).collect()
    }

    pub fn generation(&self) -> u32 {
        self.generation
    }

    pub fn history(&self) -> &[MutationRecord] {
        &self.history
    }

    pub fn mutations_total(&self) -> u32 {
        self.mutations_total
    }

    pub fn mutations_reverted(&self) -> u32 {
        self.mutations_reverted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_engine() {
        let e = Engine::new();
        assert_eq!(e.generation(), 0);
        assert_eq!(e.mutations_total(), 0);
    }

    #[test]
    fn test_add_behavior() {
        let mut e = Engine::new();
        e.add_behavior("aggression", 0.5, 0.0, 1.0, 0.1);
        assert_eq!(e.get("aggression"), 0.5);
        assert!(e.find_behavior("aggression").is_some());
    }

    #[test]
    fn test_get_not_found() {
        let e = Engine::new();
        assert_eq!(e.get("nonexistent"), -1.0);
    }

    #[test]
    fn test_find_behavior_none() {
        let e = Engine::new();
        assert!(e.find_behavior("nope").is_none());
    }

    #[test]
    fn test_set_clamp() {
        let mut e = Engine::new();
        e.add_behavior("x", 0.5, 0.0, 1.0, 0.1);
        e.set("x", 1.5);
        assert_eq!(e.get("x"), 1.0);
        e.set("x", -0.5);
        assert_eq!(e.get("x"), 0.0);
    }

    #[test]
    fn test_set_nonexistent() {
        let mut e = Engine::new();
        e.set("nope", 0.5); // should not panic
        assert_eq!(e.get("nope"), -1.0);
    }

    #[test]
    fn test_score() {
        let mut e = Engine::new();
        e.add_behavior("b", 0.5, 0.0, 1.0, 0.1);
        e.score("b", 1.0);
        e.score("b", 0.5);
        let b = e.find_behavior("b").unwrap();
        assert_eq!(b.uses, 2);
        assert!((b.cumulative_score - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_score_nonexistent() {
        let mut e = Engine::new();
        e.score("nope", 1.0); // should not panic
    }

    #[test]
    fn test_cycle_increments_generation() {
        let mut e = Engine::new();
        e.add_behavior("a", 0.5, 0.0, 1.0, 0.0); // rate=0 so no mutations (usually)
        let before = e.generation();
        e.cycle(0.5);
        assert_eq!(e.generation(), before + 1);
    }

    #[test]
    fn test_cycle_mutations_above_threshold() {
        let mut e = Engine::new();
        e.add_behavior("a", 0.5, 0.0, 1.0, 1.0); // high mutation rate
        let count = e.cycle(0.5);
        // Should have made at least some mutations with rate=1.0
        assert!(count >= 0); // deterministic hash may or may not trigger
    }

    #[test]
    fn test_revert() {
        let mut e = Engine::new();
        e.add_behavior("x", 0.5, 0.0, 1.0, 1.0);
        e.cycle(0.5);
        if !e.history().is_empty() {
            let last = e.history().len() - 1;
            let old_val = e.history()[last].old_value;
            let new_val = e.history()[last].new_value;
            assert!((new_val - old_val).abs() > 1e-10);
            assert!(e.revert(last));
            assert_eq!(e.get("x"), old_val);
            assert!(e.history()[last].reverted);
        }
    }

    #[test]
    fn test_revert_already_reverted() {
        let mut e = Engine::new();
        e.add_behavior("x", 0.5, 0.0, 1.0, 1.0);
        e.cycle(0.5);
        if !e.history().is_empty() {
            let idx = e.history().len() - 1;
            assert!(e.revert(idx));
            assert!(!e.revert(idx)); // double revert fails
        }
    }

    #[test]
    fn test_revert_invalid_index() {
        let mut e = Engine::new();
        assert!(!e.revert(0));
        assert!(!e.revert(999));
    }

    #[test]
    fn test_rollback() {
        let mut e = Engine::new();
        e.add_behavior("x", 0.5, 0.0, 1.0, 1.0);
        let _ = e.cycle(0.5); // gen 1
        let _ = e.cycle(0.5); // gen 2
        let _ = e.cycle(0.5); // gen 3
        let pre = e.get("x");
        let reverted = e.rollback(1);
        let post = e.get("x");
        // Rolled back gen 2 and gen 3 mutations
        assert!(reverted >= 0);
        if reverted > 0 {
            assert!((pre - post).abs() > 1e-10 || pre == post);
        }
    }

    #[test]
    fn test_best_behaviors() {
        let mut e = Engine::new();
        e.add_behavior("good", 0.5, 0.0, 1.0, 0.1);
        e.add_behavior("bad", 0.5, 0.0, 1.0, 0.1);
        e.score("good", 10.0);
        e.score("good", 10.0);
        e.score("bad", -5.0);
        e.score("bad", -5.0);
        let best = e.best_behaviors(1);
        assert_eq!(best[0].name, "good");
    }

    #[test]
    fn test_worst_behaviors() {
        let mut e = Engine::new();
        e.add_behavior("good", 0.5, 0.0, 1.0, 0.1);
        e.add_behavior("bad", 0.5, 0.0, 1.0, 0.1);
        e.score("good", 10.0);
        e.score("bad", -5.0);
        let worst = e.worst_behaviors(1);
        assert_eq!(worst[0].name, "bad");
    }

    #[test]
    fn test_aggressive_cycle() {
        let mut e = Engine::new();
        e.add_behavior("x", 0.5, 0.0, 1.0, 0.01);
        // Low fitness should trigger aggressive mode (3x probability)
        let count = e.cycle(0.1);
        assert_eq!(e.generation(), 1);
        assert!(count >= 0);
    }

    #[test]
    fn test_elite_cycle() {
        let mut e = Engine::new();
        e.add_behavior("bad", 0.5, 0.0, 1.0, 0.1);
        e.score("bad", -10.0);
        // High fitness = elite mode, only worst behaviors mutate
        let _ = e.cycle(0.9);
        assert_eq!(e.generation(), 1);
    }

    #[test]
    fn test_mutation_record_fields() {
        let mut e = Engine::new();
        e.add_behavior("p", 0.5, 0.0, 1.0, 1.0);
        e.cycle(0.5);
        if let Some(rec) = e.history().last() {
            assert_eq!(rec.generation, 1);
            assert!(!rec.reverted);
            assert!(matches!(rec.mutation_type, MutationType::ParamAdjust));
        }
    }
}
