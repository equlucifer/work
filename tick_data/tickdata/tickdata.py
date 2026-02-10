import io
import zipfile
import numpy as np
import pandas as pd
from xlib import xcredfile, xgcs


def signed_data_bucket_name():
    return 'lcm-data-tickdata-td-eq-us-signed-use4'


def signed_trades_object_name(tkr, date):
    return 'signed/US_TED/{0}/{1}/{2}/{3}.parquet.gzip'.format(date[:4], date[5:7], date[8:10], tkr)


def nbbo_object_name(tkr, date):
    return 'nbbo/US_TED/{0}/{1}/{2}/{3}.parquet.gzip'.format(date[:4], date[5:7], date[8:10], tkr)


def daily_data_object_name(date):
    return 'analysis/flow/daily/{0}.parquet.gzip'.format(date.replace('-', ''))


def minutely_data_object_name(tkr, date):
    return 'analysis/flow/minutely/{0}/{1}.parquet.gzip'.format(date.replace('-', ''), tkr)


def column_name_map():
    column_names = ['date', 'time', 'px', 'qty', 'venue', 'condition', 'correction', 'seq', 'trade_stop', 'source',
                    'trf', 'exclude', 'filtered_px', 'e_time', 'id', 'trf_time', 'time_obsolete',
                    'participant_time_obsolete', 'trf_time_obsolete', 'trade_through_exempt']
    col_map = dict(zip(range(len(column_names)), column_names))
    return col_map


def original_column_name_map():
    orig_names = ['Date', 'Time', 'Price', 'Volume', 'Exchange Code', 'Sales Condition', 'Correction Indicator',
                  'Sequence Number', 'Trade Stop Indicator', 'Source of Trade',
                  'MDS 127 / TRF (Trade Reporting Facility)', 'Exclude Record Flag', 'Filtered Price',
                  'Participant Timestamp', 'Trade ID', 'Trade Reporting Facility (TRF) Timestamp',
                  'Timestamp Microseconds (Obsolete as of 10/24/2016)',
                  'Participant Timestamp Microseconds (Obsolete as of 10/24/2016)',
                  'TRF Timestamp Microseconds (Obsolete as of 10/24/2016)', 'Trade Through Exempt Indicator']
    current_names = ['date', 'time', 'px', 'qty', 'venue', 'condition', 'correction', 'seq', 'trade_stop', 'source',
                     'trf', 'exclude', 'filtered_px', 'e_time', 'id', 'trf_time', 'time_obsolete',
                     'participant_time_obsolete', 'trf_time_obsolete', 'trade_through_exempt']
    col_remap = dict(zip(current_names, orig_names))
    return col_remap


def load_quotes(tkr, date, column_map=None, zipname=None, dtype_map=None, include_sizes=False, gc_load=False):
    """
    include_sizes := will be ignored if column_map is not None
    gc_load := if True, load quotes from google cloud
    zipname := this parameter is ignored when gc_load == True
    """
    if column_map is None:
        if include_sizes:
            cols_old = [0, 1, 2, 3, 4, 5, 6, 23]
            cols_new = ['date', 'time', 'venue', 'bid', 'ask', 'qty_bid', 'qty_ask', 'e_time']
        else:
            cols_old = [0, 1, 2, 3, 4, 23]
            cols_new = ['date', 'time', 'venue', 'bid', 'ask', 'e_time']
        if date < '2015-09-01':
            cols_old = cols_old[:-1]
            cols_new = cols_new[:-1]
    else:
        cols_old = list(column_map.keys())
        cols_new = list(column_map.values())
    if gc_load:
        bucket = xgcs.make_conn('lcm-data-tickdata-use4')
        oname = f'data/US_TED/QUOTES/{date[:4]}/{date[5:7]}/{date[8:10]}/{tkr}.zip'
        buffer = io.BytesIO(b'')
        bucket.downloadToFileObject(oname, buffer)
        with zipfile.ZipFile(buffer, 'r') as zf:
            with zf.open(zf.filelist[0].filename) as file:
                df = pd.read_csv(file, header=None, usecols=cols_old, dtype=dtype_map)
    else:
        year = date[:4]
        month = date[5:7]
        date_str = date.replace('-', '_')
        if zipname is None:
            zipname = '/data/tickdata/data/US_TED/{0}/{1}/QUOTES/{2}_X_Q_{3}.zip'.format(year, month, date_str, tkr[0])
        archive = zipfile.ZipFile(zipname, 'r')
        subfolder = '{0}_{1}_X_Q.zip'.format(tkr, date_str)
        fname = '{0}_{1}_X_Q.asc'.format(tkr, date_str)
        subarchive = zipfile.ZipFile(archive.open(subfolder), 'r')
        df = pd.read_csv(subarchive.open(fname), header=None, usecols=cols_old, dtype=dtype_map)
    df.columns = cols_new
    return df


def load_trades(tkr, date, column_map=None, zipname=None, dtype_map=None):
    if column_map is None:
        new_cols = ['date', 'time', 'px', 'qty', 'venue', 'condition', 'correction', 'seq', 'trade_stop', 'source',
                    'trf', 'exclude', 'filtered_px', 'e_time', 'id', 'trf_time', 'time_obsolete',
                    'participant_time_obsolete', 'trf_time_obsolete', 'trade_through_exempt']
        column_map = dict(zip(range(len(new_cols)), new_cols))
        usecols = None
    else:
        usecols = sorted(column_map.keys())
    if dtype_map is None:
        dtype_map = {11: 'object'}
    year = date[:4]
    month = date[5:7]
    date_str = date.replace('-', '_')
    if zipname is None:
        zipname = '/data/tickdata/data/US_TED/{0}/{1}/TRADE/{2}_X_T.zip'.format(year, month, date_str)
    archive = zipfile.ZipFile(zipname, 'r')
    subfolder = '{0}_{1}_X_T.zip'.format(tkr, date_str)
    fname = '{0}_{1}_X_T.asc'.format(tkr, date_str)
    subarchive = zipfile.ZipFile(archive.open(subfolder), 'r')
    df = pd.read_csv(subarchive.open(fname), header=None, usecols=usecols, dtype=dtype_map)
    df = df.rename(columns=column_map)
    return df


def sign_trades(tkr, date, dfq=None, dft=None, ts_column=None, ts_column_old=None, zipname_quotes=None,
                zipname_trades=None):
    """
    note: sign in two ways:
        (i) using nbbo
        (ii) using own venue's bid/ask (except when not available, when using nbbo)
    note: also saves bids/asks (both nbbo and own venue)
    note: if dfq is not None, then zipname_quotes is ignored; similarly, if dft is not None, then zipname_trades is
        ignored
    """
    if ts_column is None:
        ts_column = 'e_time'
    if ts_column_old is None:
        ts_column_old = 'time'
    if dfq is None:
        dfq = load_quotes(tkr, date, zipname=zipname_quotes)
    if dft is None:
        dft = load_trades(tkr, date, zipname=zipname_trades)
    ts_column_use = ts_column
    if date < '2015-09-01':
        ts_column_use = ts_column_old
    dft.index = dft[ts_column_use].copy()
    dft['tag'] = range(len(dft))  # to allow to sort at the end in the original order
    dfq['bid'] = dfq['bid'].replace(0, -np.inf)
    dfq['ask'] = dfq['ask'].replace(0, np.inf)
    df_bid = dfq.pivot_table(index=ts_column_use, columns='venue', values='bid', aggfunc='last').fillna(method='ffill')
    df_bid['nbbo'] = df_bid.max(axis=1)
    df_ask = dfq.pivot_table(index=ts_column_use, columns='venue', values='ask', aggfunc='last').fillna(method='ffill')
    df_ask['nbbo'] = df_ask.min(axis=1)
    df_bid = df_bid.reindex(df_bid.index.union(dft.index).drop_duplicates()).sort_index().fillna(method=
                                                                                        'ffill').shift(1)
    df_ask = df_ask.reindex(df_bid.index).fillna(method='ffill').shift(1)
    dft['nbbo_bid'] = df_bid.loc[dft.index, 'nbbo']
    dft['nbbo_ask'] = df_ask.loc[dft.index, 'nbbo']
    s_diff = (dft['px'] - (dft['nbbo_bid'] + dft['nbbo_ask']) / 2).round(10)
    s_diff = s_diff.mask(dft['nbbo_bid'] == -np.inf, (dft['px'] >= dft['nbbo_ask']).astype(int))
    s_diff = s_diff.mask(dft['nbbo_ask'] == np.inf, -(dft['px'] <= dft['nbbo_bid']).astype(int))
    dft['nbbo_sign'] = np.sign(s_diff)
    dft['venue_bid'] = np.nan
    dft['venue_ask'] = np.nan
    dft['venue_sign'] = np.nan
    df_list = []
    for venue, dft_venue in dft.groupby('venue'):
        if venue not in df_bid.columns:
            s_bid = df_bid.loc[dft_venue.index, 'nbbo']
        else:
            s_bid = df_bid.loc[dft_venue.index, venue]
        dft_venue['venue_bid'] = s_bid
        if venue not in df_ask.columns:
            s_ask = df_ask.loc[dft_venue.index, 'nbbo']
        else:
            s_ask = df_ask.loc[dft_venue.index, venue]
        dft_venue['venue_ask'] = s_ask
        s_diff = (dft_venue['px'] - (s_bid + s_ask) / 2).round(10)
        s_diff = s_diff.mask(s_bid == -np.inf, (dft_venue['px'] >= s_ask).astype(int))
        s_diff = s_diff.mask(s_ask == np.inf, -(dft_venue['px'] <= s_bid).astype(int))
        dft_venue['venue_sign'] = np.sign(s_diff)
        df_list.append(dft_venue.copy())
    df = pd.concat(df_list)
    df = df.set_index(['tag']).sort_index().copy()
    df.index.name = None
    return df


def get_signed_trades(tkr, date, bucket=None, keep_original_names=False):
    """
    keep_original_names := will use original tickdata names for columns
    """
    if bucket is None:
        bucket = xgcs.make_conn(signed_data_bucket_name())
    buffer = io.BytesIO(b'')
    object_name = signed_trades_object_name(tkr, date)
    bucket.downloadToFileObject(object_name, buffer)
    df = pd.read_parquet(buffer)
    if keep_original_names:
        df = df.rename(columns=original_column_name_map())
    return df


def calc_nbbo(tkr, date, dfq=None, ts_column=None, ts_column_old=None, zipname=None, return_detail=False):
    """
    note: if dfq is not None, then zipname is ignored
    """
    if ts_column is None:
        ts_column = 'e_time'
    if ts_column_old is None:
        ts_column_old = 'time'
    if dfq is None:
        dfq = load_quotes(tkr, date, zipname=zipname, include_sizes=True)
    ts_column_use = ts_column
    if date < '2015-09-01':
        ts_column_use = ts_column_old
    dfq['bid'] = dfq['bid'].replace(0, -np.inf)
    dfq['ask'] = dfq['ask'].replace(0, np.inf)
    df_bid = dfq.pivot_table(index=ts_column_use, columns='venue', values='bid', aggfunc='last',
                             dropna=False).fillna(method='ffill')
    df_ask = dfq.pivot_table(index=ts_column_use, columns='venue', values='ask', aggfunc='last',
                             dropna=False).fillna(method='ffill')
    df_qty_bid = dfq.pivot_table(index=ts_column_use, columns='venue', values='qty_bid', aggfunc='last',
                                 dropna=False).fillna(method='ffill')
    df_qty_ask = dfq.pivot_table(index=ts_column_use, columns='venue', values='qty_ask', aggfunc='last',
                                 dropna=False).fillna(method='ffill')
    s_bid_nbbo = df_bid.max(axis=1)
    s_ask_nbbo = df_ask.min(axis=1)
    s_qty_bid_nbbo = (df_bid.apply(lambda x: x == s_bid_nbbo).astype(int) * df_qty_bid).sum(axis=1)
    s_qty_ask_nbbo = (df_ask.apply(lambda x: x == s_ask_nbbo).astype(int) * df_qty_ask).sum(axis=1)
    df = pd.concat([s_bid_nbbo, s_qty_bid_nbbo, s_ask_nbbo, s_qty_ask_nbbo], axis=1,
                    keys=['bid', 'qty_bid', 'ask', 'qty_ask'], sort=True)
    df = df.reset_index().copy()
    if return_detail:
        out = dict()
        out['nbbo'] = df.copy()
        out['bid'] = df_bid.copy()
        out['ask'] = df_ask.copy()
        out['qty_bid'] = df_qty_bid.copy()
        out['qty_ask'] = df_qty_ask.copy()
    else:
        out = df.copy()
    return out


def load_nbbo(tkr, date, bucket=None):
    """
    load pre-saved NBBO
    """
    if bucket is None:
        bucket = xgcs.make_conn(signed_data_bucket_name())
    buffer = io.BytesIO(b'')
    object_name = nbbo_object_name(tkr, date)
    bucket.downloadToFileObject(object_name, buffer)
    df = pd.read_parquet(buffer)
    return df


def get_price_improved_trades(ticker, dt):
    col_map = dict({5: 'condition', 11: 'exclude'})
    dtype_map = {11: 'object'}
    eps = 1e-9
    allowed_conditions = ['I', 'T', 'TI', '@', '@ I', '@ T', '@ TI']  # note: blank condition appears as na

    df = get_signed_trades(ticker, dt)
    p = df.px % 0.01
    is_price_improved = ((p > eps) & (p < 0.005 - eps)) | ((p < 0.01 - eps) & (p > 0.005 + eps))
    df = df[(df.exclude != 'X') & (df.venue == 'D') & (
            df.condition.isna() | df.condition.isin(allowed_conditions)) & is_price_improved & (df.px > 1)]
    return df


def get_price_improved_buckets(ticker, dt, bucketing = '5min'):
    df = get_price_improved_trades(ticker, dt)
    df['e_time'] = pd.to_datetime(df.date + ' ' + df.e_time, format = '%m/%d/%Y %H:%M:%S.%f')
    df['notional'] = df.px * df.qty
    # venue_sign is NBBO sign for dark
    return df.groupby([pd.Grouper(key = 'e_time', freq = bucketing), 'venue_sign']).notional.sum().unstack()
