
from utils import *


batch_size = 500
enc_seq_len = 6
output_sequence_length = 1


data_loader = custom_loader(batch_size, './data/container_cpu_usage_seconds_total.csv')

data_loader.get_next_batch_and_label(enc_seq_len, output_sequence_length)
data_loader.get_next_batch_and_label(enc_seq_len, output_sequence_length)
data_loader.get_next_batch_and_label(enc_seq_len, output_sequence_length)
data_loader.get_next_batch_and_label(enc_seq_len, output_sequence_length)
data_loader.get_next_batch_and_label(enc_seq_len, output_sequence_length)
data_loader.get_next_batch_and_label(enc_seq_len, output_sequence_length)

# X, Y = get_data(batch_size, enc_seq_len, output_sequence_length)

