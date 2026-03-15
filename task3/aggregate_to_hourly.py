import pandas as pd
import cudf

agg_dict = {
    'x1': ['mean', 'max'],
    'x2': 'mean'
}
temp_cols = [f't{i}' for i in range(1, 14)]
for col in temp_cols:
    agg_dict[col] = 'mean'

chunk_list = []
chunk_size = 1000000 

for i, chunk_pd in enumerate(pd.read_csv('/content/data.csv', chunksize=chunk_size)):
    chunk_pd['timedate'] = pd.to_datetime(chunk_pd['timedate'], utc=True).dt.tz_localize(None)
    chunk_gpu = cudf.from_pandas(chunk_pd)
    chunk_gpu['hour'] = chunk_gpu['timedate'].dt.floor('H')
    partial_agg_gpu = chunk_gpu.groupby(['deviceId', 'hour']).agg(agg_dict)
    chunk_list.append(partial_agg_gpu.to_pandas())


consolidated_df = pd.concat(chunk_list).groupby(level=['deviceId', 'hour']).agg({
    ('x1', 'mean'): 'mean',
    ('x1', 'max'): 'max',
    ('x2', 'mean'): 'mean',
    **{ (col, 'mean'): 'mean' for col in temp_cols }
}).reset_index()

consolidated_df.columns = [('_'.join(col).strip('_') if isinstance(col, tuple) else col) for col in consolidated_df.columns]

devices_df = pd.read_csv('/content/devices.csv')
final_df = pd.merge(consolidated_df, devices_df, on='deviceId', how='left')

output_path = '/content/dane_task3_aggregated.csv.gz'
final_df.to_csv(output_path, index=False, compression='gzip')

display(final_df.head())