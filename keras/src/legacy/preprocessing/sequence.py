"""Deprecated sequence preprocessing APIs from Keras 1."""

import json
import random

import numpy as np

from keras.src.api_export import keras_export
from keras.src.trainers.data_adapters.py_dataset_adapter import PyDataset


@keras_export("keras._legacy.preprocessing.sequence.TimeseriesGenerator")
class TimeseriesGenerator(PyDataset):
    """Utility class for generating batches of temporal data.

    DEPRECATED.

    This class takes in a sequence of data-points gathered at
    equal intervals, along with time series parameters such as
    stride, length of history, etc., to produce batches for
    training/validation.

    Arguments:
        data: Indexable generator (such as list or Numpy array)
            containing consecutive data points (timesteps).
            The data should be at 2D, and axis 0 is expected
            to be the time dimension.
        targets: Targets corresponding to timesteps in `data`.
            It should have same length as `data`.
        length: Length of the output sequences (in number of timesteps).
        sampling_rate: Period between successive individual timesteps
            within sequences. For rate `r`, timesteps
            `data[i]`, `data[i-r]`, ... `data[i - length]`
            are used for create a sample sequence.
        stride: Period between successive output sequences.
            For stride `s`, consecutive output samples would
            be centered around `data[i]`, `data[i+s]`, `data[i+2*s]`, etc.
        start_index: Data points earlier than `start_index` will not be used
            in the output sequences. This is useful to reserve part of the
            data for test or validation.
        end_index: Data points later than `end_index` will not be used
            in the output sequences. This is useful to reserve part of the
            data for test or validation.
        shuffle: Whether to shuffle output samples,
            or instead draw them in chronological order.
        reverse: Boolean: if `true`, timesteps in each output sample will be
            in reverse chronological order.
        batch_size: Number of timeseries samples in each batch
            (except maybe the last one).
        **kwargs: Additional keyword arguments for the `PyDataset` base class,
            such as `workers`, `use_multiprocessing`, and `max_queue_size`.

    Returns:
        A PyDataset instance.
    """

    def __init__(
        self,
        data,
        targets,
        length,
        sampling_rate=1,
        stride=1,
        start_index=0,
        end_index=None,
        shuffle=False,
        reverse=False,
        batch_size=128,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if len(data) != len(targets):
            raise ValueError(
                "Data and targets have to be "
                f"of same length. Data length is {len(data)} "
                f"while target length is {len(targets)}"
            )

        self.data = data
        self.targets = targets
        self.length = length
        self.sampling_rate = sampling_rate
        self.stride = stride
        self.start_index = start_index + length
        if end_index is None:
            end_index = len(data) - 1
        self.end_index = end_index
        self.shuffle = shuffle
        self.reverse = reverse
        self.batch_size = batch_size

        if self.start_index > self.end_index:
            raise ValueError(
                f"`start_index+length={self.start_index} "
                f"> end_index={self.end_index}` "
                "is disallowed, as no part of the sequence "
                "would be left to be used as current step."
            )

    def __len__(self):
        return (
            self.end_index - self.start_index + self.batch_size * self.stride
        ) // (self.batch_size * self.stride)

    def __getitem__(self, index):
        if self.shuffle:
            rows = np.random.randint(
                self.start_index, self.end_index + 1, size=self.batch_size
            )
        else:
            i = self.start_index + self.batch_size * self.stride * index
            rows = np.arange(
                i,
                min(i + self.batch_size * self.stride, self.end_index + 1),
                self.stride,
            )

        samples = np.array(
            [
                self.data[row - self.length : row : self.sampling_rate]
                for row in rows
            ]
        )
        targets = np.array([self.targets[row] for row in rows])

        if self.reverse:
            return samples[:, ::-1, ...], targets
        return samples, targets

    def get_config(self):
        """Returns the TimeseriesGenerator configuration as Python dictionary.

        Returns:
            A Python dictionary with the TimeseriesGenerator configuration.
        """
        data = self.data
        if type(self.data).__module__ == np.__name__:
            data = self.data.tolist()
        try:
            json_data = json.dumps(data)
        except TypeError as e:
            raise TypeError(f"Data not JSON Serializable: {data}") from e

        targets = self.targets
        if type(self.targets).__module__ == np.__name__:
            targets = self.targets.tolist()
        try:
            json_targets = json.dumps(targets)
        except TypeError as e:
            raise TypeError(f"Targets not JSON Serializable: {targets}") from e

        config = super().get_config()
        config.update(
            {
                "data": json_data,
                "targets": json_targets,
                "length": self.length,
                "sampling_rate": self.sampling_rate,
                "stride": self.stride,
                "start_index": self.start_index,
                "end_index": self.end_index,
                "shuffle": self.shuffle,
                "reverse": self.reverse,
                "batch_size": self.batch_size,
            }
        )
        return config

    def to_json(self, **kwargs):
        """Returns a JSON string containing the generator's configuration.

        Args:
            **kwargs: Additional keyword arguments to be passed
                to `json.dumps()`.

        Returns:
            A JSON string containing the tokenizer configuration.
        """
        config = self.get_config()
        timeseries_generator_config = {
            "class_name": self.__class__.__name__,
            "config": config,
        }
        return json.dumps(timeseries_generator_config, **kwargs)


@keras_export("keras._legacy.preprocessing.sequence.make_sampling_table")
def make_sampling_table(size, sampling_factor=1e-5):
    """Generates a word rank-based probabilistic sampling table.

    DEPRECATED.

    Used for generating the `sampling_table` argument for `skipgrams`.
    `sampling_table[i]` is the probability of sampling
    the word i-th most common word in a dataset
    (more common words should be sampled less frequently, for balance).

    The sampling probabilities are generated according
    to the sampling distribution used in word2vec:

    ```
    p(word) = (min(1, sqrt(word_frequency / sampling_factor) /
        (word_frequency / sampling_factor)))
    ```

    We assume that the word frequencies follow Zipf's law (s=1) to derive
    a numerical approximation of frequency(rank):

    `frequency(rank) ~ 1/(rank * (log(rank) + gamma) + 1/2 - 1/(12*rank))`
    where `gamma` is the Euler-Mascheroni constant.

    Args:
        size: Int, number of possible words to sample.
        sampling_factor: The sampling factor in the word2vec formula.

    Returns:
        A 1D Numpy array of length `size` where the ith entry
        is the probability that a word of rank i should be sampled.
    """
    gamma = 0.577
    rank = np.arange(size)
    rank[0] = 1
    inv_fq = rank * (np.log(rank) + gamma) + 0.5 - 1.0 / (12.0 * rank)
    f = sampling_factor * inv_fq

    return np.minimum(1.0, f / np.sqrt(f))


@keras_export("keras._legacy.preprocessing.sequence.skipgrams")
def skipgrams(
    sequence,
    vocabulary_size,
    window_size=4,
    negative_samples=1.0,
    shuffle=True,
    categorical=False,
    sampling_table=None,
    seed=None,
):
    """Generates skipgram word pairs.

    DEPRECATED.

    This function transforms a sequence of word indexes (list of integers)
    into tuples of words of the form:

    - (word, word in the same window), with label 1 (positive samples).
    - (word, random word from the vocabulary), with label 0 (negative samples).

    Read more about Skipgram in this gnomic paper by Mikolov et al.:
    [Efficient Estimation of Word Representations in
    Vector Space](http://arxiv.org/pdf/1301.3781v3.pdf)

    Args:
        sequence: A word sequence (sentence), encoded as a list
            of word indices (integers). If using a `sampling_table`,
            word indices are expected to match the rank
            of the words in a reference dataset (e.g. 10 would encode
            the 10-th most frequently occurring token).
            Note that index 0 is expected to be a non-word and will be skipped.
        vocabulary_size: Int, maximum possible word index + 1
        window_size: Int, size of sampling windows (technically half-window).
            The window of a word `w_i` will be
            `[i - window_size, i + window_size+1]`.
        negative_samples: Float >= 0. 0 for no negative (i.e. random) samples.
            1 for same number as positive samples.
        shuffle: Whether to shuffle the word couples before returning them.
        categorical: bool. if False, labels will be
            integers (eg. `[0, 1, 1 .. ]`),
            if `True`, labels will be categorical, e.g.
            `[[1,0],[0,1],[0,1] .. ]`.
        sampling_table: 1D array of size `vocabulary_size` where the entry i
            encodes the probability to sample a word of rank i.
        seed: Random seed.

    Returns:
        couples, labels: where `couples` are int pairs and
            `labels` are either 0 or 1.

    Note:
        By convention, index 0 in the vocabulary is
        a non-word and will be skipped.
    """

    # Precompute valid indices to avoid repeated computation
    sequence_len = len(sequence)
    # Preallocate lists for memory efficiency
    couples = []
    labels = []

    # Use local function lookup to reduce global lookups in tight loop
    append_couple = couples.append
    append_label = labels.append
    min_ = min
    max_ = max

    # Generate positive samples
    for i, wi in enumerate(sequence):
        if not wi:
            continue
        if sampling_table is not None:
            if sampling_table[wi] < random.random():
                continue
        # Use local range bounds helpers
        window_start = max_(0, i - window_size)
        window_end = min_(sequence_len, i + window_size + 1)
        for j in range(window_start, window_end):
            if j == i:
                continue
            wj = sequence[j]
            if not wj:
                continue
            append_couple([wi, wj])
            if categorical:
                append_label([0, 1])
            else:
                append_label(1)

    # Generate negative samples
    if negative_samples > 0:
        num_positive = len(labels)
        num_negative_samples = int(num_positive * negative_samples)
        if num_positive > 0:
            words = [couple[0] for couple in couples]
            random.shuffle(words)
            rand_word = random.randint
            vocab_range = (1, vocabulary_size - 1)
            append_couple_neg = couples.append
            # Use local append functions and avoid repeated computes
            neg_label = [1, 0] if categorical else 0
            for i in range(num_negative_samples):
                w0 = words[i % num_positive]
                w1 = rand_word(*vocab_range)
                append_couple_neg([w0, w1])
                if categorical:
                    append_label([1, 0])
                else:
                    append_label(0)
        else:
            # No positives to mirror for negatives; do nothing (original behavior)
            pass

    # Shuffle if needed

    if shuffle:
        actual_seed = seed if seed is not None else random.randint(0, int(10e6))
        # Use one shuffled index to permute both lists for guaranteed same order
        # This is faster and memory efficient than shuffling two large lists independently
        data = list(zip(couples, labels))
        random.seed(actual_seed)
        random.shuffle(data)
        # Unzip back
        if data:
            couples, labels = zip(*data)
            couples = list(couples)
            labels = list(labels)
        else:
            couples, labels = [], []

    return couples, labels
