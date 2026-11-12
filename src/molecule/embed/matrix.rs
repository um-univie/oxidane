//! A dense square matrix, row-major, for the distance bounds and the
//! metric-matrix step of the embedding -- the little of a matrix library
//! they need.

use core::ops::{Index, IndexMut};

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SquareMatrix<T> {
    values: Vec<T>,
    size: usize,
}

impl<T: Clone> SquareMatrix<T> {
    /// A `size x size` matrix with every entry `value`.
    pub(crate) fn from_elem(size: usize, value: T) -> Self {
        Self { values: vec![value; size * size], size }
    }

    /// The matrix with `f` applied to every entry.
    pub(crate) fn map<U>(&self, f: impl Fn(&T) -> U) -> SquareMatrix<U> {
        SquareMatrix { values: self.values.iter().map(f).collect(), size: self.size }
    }
}

impl SquareMatrix<f64> {
    pub(crate) fn zeros(size: usize) -> Self {
        Self::from_elem(size, 0.0)
    }
}

impl<T> SquareMatrix<T> {
    /// The number of rows, which is also the number of columns.
    pub(crate) fn nrows(&self) -> usize {
        self.size
    }

    pub(crate) fn row(&self, row: usize) -> &[T] {
        &self.values[row * self.size..(row + 1) * self.size]
    }
}

impl<T> Index<[usize; 2]> for SquareMatrix<T> {
    type Output = T;

    #[inline]
    fn index(&self, [row, column]: [usize; 2]) -> &T {
        assert!(column < self.size, "column {column} out of {}", self.size);
        &self.values[row * self.size + column]
    }
}

impl<T> IndexMut<[usize; 2]> for SquareMatrix<T> {
    #[inline]
    fn index_mut(&mut self, [row, column]: [usize; 2]) -> &mut T {
        assert!(column < self.size, "column {column} out of {}", self.size);
        &mut self.values[row * self.size + column]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexes_row_major() {
        let mut matrix = SquareMatrix::zeros(3);
        matrix[[1, 2]] = 5.0;
        matrix[[2, 0]] = 7.0;
        assert_eq!(matrix.row(1), &[0.0, 0.0, 5.0]);
        assert_eq!(matrix.row(2), &[7.0, 0.0, 0.0]);
        assert_eq!(matrix.map(|v| v * 2.0)[[2, 0]], 14.0);
        assert_eq!(matrix.nrows(), 3);
    }

    #[test]
    #[should_panic]
    fn rejects_a_column_past_the_edge() {
        let matrix = SquareMatrix::<f64>::zeros(3);
        let _ = matrix[[0, 3]];
    }
}
