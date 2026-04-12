#![allow(dead_code)]
pub struct Genome { id: u32, genes: Vec<f64>, fitness: f64, generation: u32 }
pub struct Evolve { population: Vec<Genome>, next_id: u32, generation: u32, mutation_rate: f64, elite_count: usize }
impl Evolve {
    pub fn new(pop_size: usize, gene_count: usize, mutation_rate: f64) -> Self {
        let mut e = Self { population: Vec::new(), next_id: 1, generation: 0, mutation_rate, elite_count: (pop_size / 5).max(1) };
        for _ in 0..pop_size {
            let id = e.next_id; e.next_id += 1;
            e.population.push(Genome { id, genes: (0..gene_count).map(|_| rand::random::<f64>()).collect(), fitness: 0.0, generation: 0 });
        } e
    }
    pub fn evaluate(&mut self, fitnesses: &[f64]) { for (i, f) in fitnesses.iter().enumerate().take(self.population.len()) { self.population[i].fitness = *f; } }
    pub fn select(&mut self) { self.population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap()); let n = self.population.len(); self.population.truncate(self.elite_count.min(n)); }
    pub fn crossover(&mut self) {
        let elite: Vec<Vec<f64>> = self.population.iter().map(|g| g.genes.clone()).collect(); let target = (self.population.len() * 3).max(10);
        while self.population.len() < target {
            if elite.len() < 2 { break; }
            let (p1, p2) = (&elite[rand::random::<usize>() % elite.len()], &elite[rand::random::<usize>() % elite.len()]);
            let child_genes: Vec<f64> = p1.iter().zip(p2.iter()).map(|(a, b)| if rand::random::<f64>() < 0.5 { *a } else { *b }).collect();
            let id = self.next_id; self.next_id += 1;
            self.population.push(Genome { id, genes: child_genes, fitness: 0.0, generation: self.generation });
        }
    }
    pub fn mutate(&mut self) { for g in &mut self.population { for gene in &mut g.genes { if rand::random::<f64>() < self.mutation_rate { *gene += (rand::random::<f64>() - 0.5) * 0.1; } } } }
    pub fn best(&self) -> Option<&Genome> { self.population.iter().max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap()) }
    pub fn worst(&self) -> Option<&Genome> { self.population.iter().min_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap()) }
    pub fn average_fitness(&self) -> f64 { if self.population.is_empty() { 0.0 } else { self.population.iter().map(|g| g.fitness).sum::<f64>() / self.population.len() as f64 } }
    pub fn population_size(&self) -> usize { self.population.len() }
    pub fn generation(&self) -> u32 { self.generation }
    pub fn step(&mut self, fitnesses: &[f64]) { self.evaluate(fitnesses); self.select(); self.crossover(); self.mutate(); self.generation += 1; }
    pub fn inject(&mut self, genes: Vec<f64>) { let id = self.next_id; self.next_id += 1; self.population.push(Genome { id, genes, fitness: 0.0, generation: self.generation }); }
    pub fn diversity(&self) -> f64 {
        if self.population.is_empty() { return 0.0; }
        let n = self.population[0].genes.len(); if n == 0 { return 0.0; }
        let means: Vec<f64> = (0..n).map(|i| self.population.iter().map(|g| g.genes[i]).sum::<f64>() / self.population.len() as f64).collect();
        means.iter().map(|&m| self.population.iter().map(|g| g.genes.iter().map(|&gene| (gene - m).powi(2)).sum::<f64>()).sum::<f64>()).sum::<f64>() / (self.population.len() * n) as f64
    }
    pub fn set_mutation_rate(&mut self, rate: f64) { self.mutation_rate = rate; }
    pub fn genes_of(&self, id: u32) -> Option<&[f64]> { self.population.iter().find(|g| g.id == id).map(|g| g.genes.as_slice()) }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn test_new() { let e = Evolve::new(10, 5, 0.1); assert_eq!(e.population_size(), 10); }
    #[test] fn test_evaluate() { let mut e = Evolve::new(5, 3, 0.1); e.evaluate(&[1.0, 0.5, 0.3, 0.8, 0.1]); assert!(e.best().unwrap().fitness > 0.8); }
    #[test] fn test_select() { let mut e = Evolve::new(10, 3, 0.1); e.evaluate(&[1.0, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]); e.select(); assert!(e.population_size() <= 3); }
    #[test] fn test_best_worst() { let mut e = Evolve::new(5, 3, 0.0); e.evaluate(&[1.0, 0.5, 0.3, 0.8, 0.1]); assert!(e.best().unwrap().fitness >= e.worst().unwrap().fitness); }
    #[test] fn test_average_fitness() { let mut e = Evolve::new(4, 3, 0.0); e.evaluate(&[1.0, 2.0, 3.0, 4.0]); assert!((e.average_fitness() - 2.5).abs() < 1e-6); }
    #[test] fn test_step() { let mut e = Evolve::new(10, 5, 0.1); e.step(&vec![1.0; 10]); assert_eq!(e.generation(), 1); }
    #[test] fn test_inject() { let mut e = Evolve::new(5, 3, 0.0); e.inject(vec![0.5, 0.5, 0.5]); assert_eq!(e.population_size(), 6); }
    #[test] fn test_diversity() { let e = Evolve::new(10, 5, 0.0); assert!(e.diversity() > 0.0); }
    #[test] fn test_genes_of() { let e = Evolve::new(5, 3, 0.0); assert!(e.genes_of(e.population[0].id).is_some()); }
    #[test] fn test_set_mutation_rate() { let mut e = Evolve::new(5, 3, 0.1); e.set_mutation_rate(0.5); assert!((e.mutation_rate - 0.5).abs() < 1e-6); }
}