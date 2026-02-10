import sys
import os
import io
import pytz
from datetime import datetime
import pandas as pd
import numpy as np
from vol_group.tickdata import tickdata
from xlib import xcredfile, xgcs
import GetData as gd
import Calendar
if not '/shared/apps/scripts' in sys.path:
    sys.path.append('/shared/apps/scripts')
import UTIL.kbook as kdb
MKT_OPEN_TIME = datetime.strptime('09:30:00.000000', '%H:%M:%S.%f').time()
MKT_CLOSE_TIME = datetime.strptime('16:00:00.000000', '%H:%M:%S.%f').time()

def prepare_data_summary_for_analysis(tkr, date, minutely=None, start_time=None, end_time=None, sign_type=None,
                                      bucket=None, df_signed=None, return_raw=None):
    """
    minutely := True or False (this will give daily data), None defaults to False, (parameter ignored when
        return_raw == True)
    start_time := default is '09:30:00.000000000'
    end_time := default is '16:00'
    sign_type := 'nbbo' or 'venue', None defaults to 'nbbo'
    note: data is collected for the interval (start_time, end_time]; i.e., start_time is excluded while end_time is
        included
    note: in order to not filter by time at all, one can set parameters, e.g., like this:
        start_time = '', end_time = '24:00'
    bucket := GC bucket where signed trades are; load from GC will be faster when this is passed
    df_signed := can pass signed data in df_signed instead of loading them from the raw signed data file; if
        df_signed is not None, bucket parameter is ignored
    return_raw := if True, return trade-by-trade, ungrouped data (in this case, minutely parameter is ignored),
        default is False
    """
    if minutely is None:
        minutely = False
    if start_time is None:
        start_time = '09:30:00.000000000'
    if end_time is None:
        end_time = '16:00'
    if minutely not in [True, False]:
        raise ValueError('minutely parameter must be True, False, or None')
    if sign_type is None:
        sign_type = 'nbbo'
    if sign_type not in ['nbbo', 'venue']:
        raise ValueError('unrecognized sign type')
    if return_raw is None:
        return_raw = False
    # raw dataframe
    if df_signed is None:
        if date == datetime.strftime(datetime.today(), '%Y-%m-%d'):
            df_raw = get_signed_trades_live(tkr, date)
        else:
            df_raw = tickdata.get_signed_trades(tkr, date, bucket=bucket)
    else:
        df_raw = df_signed.copy()
    # minutely tags
    ts_tags = sorted(['{0:02d}:{1:02d}'.format(h, m) for h in range(24) for m in range(60)])
    ts_breakpoints = [int(v[:2]) * 3600 + int(v[3:5]) * 60 for v in ts_tags]
    ts_tags = ts_tags[1:]
    # minutely dataframe
    if date < '2015-09-01':
        time_col = 'time'
    else:
        time_col = 'e_time'
    df = df_raw.copy()
    df = df[df['exclude'] != 'X'].copy()
    df = df[(df[time_col] > start_time) & (df[time_col] <= end_time)].copy()
    # make sign for auctions nan
    df.loc[df.condition.notna() & df.condition.str.contains('X'), 'nbbo_sign'] = np.nan
    df.loc[df.condition.notna() & df.condition.str.contains('X'), 'venue_sign'] = np.nan
    df['notional'] = df['px'] * df['qty']
    df['ts'] = [int(v[:2]) * 3600 + int(v[3:5]) * 60 + float(v[6:]) for v in df[time_col]]
    df['b_ts'] = pd.cut(df['ts'], ts_breakpoints, labels=ts_tags).astype(str)
    inverted_venues = ['B', 'J', 'Y', 'C']
    df['venue_type'] = df['venue'].mask(~df['venue'].isin(inverted_venues + ['D']), 'I')
    df['venue_type'] = df['venue_type'].mask(~df['venue_type'].isin(['D', 'I']), 'I')
    df['odd_lot'] = df['condition'].fillna('').str.contains('I')
    df['iso'] = df['condition'].fillna('').str.contains('F')
    eps = 1e-9
    allowed_conditions = ['I', 'T', 'TI', '@', '@ I', '@ T', '@ TI']  # note: blank condition appears as na
    p = df['px'] % 0.01
    is_price_improved = ((p > eps) & (p < 0.005 - eps)) | ((p < 0.01 - eps) & (p > 0.005 + eps))
    is_allowed_condition = (df['condition'].isna()) | df['condition'].isin(allowed_conditions)
    df['px_improved'] = (df['venue'] == 'D') & is_allowed_condition & is_price_improved & (df['px'] > 1)
    df['count'] = 1
    sign_column = '{0}_sign'.format(sign_type)
    df['sign'] = df[sign_column].copy()
    if return_raw:
        df = df.copy()
    elif minutely:
        df = df.groupby(['b_ts', 'sign', 'venue_type', 'odd_lot', 'iso', 'px_improved'],
                        as_index=False)[['count', 'qty', 'notional']].sum()
    else:
        df = df.groupby(['sign', 'venue_type', 'odd_lot', 'iso', 'px_improved'],
                        as_index=False)[['count', 'qty', 'notional']].sum()
    df['tkr'] = tkr
    df['date'] = date
    return df


def load_daily_data_summary(start_date, end_date, tkrs):
    """
    tkrs := list of tkrs or None; if None then will load all tkrs; note that loading all tickers for many dates
        can result in a large dataframe, potentially running into memory constraints
    """
    bucket = xgcs.make_conn(tickdata.signed_data_bucket_name())
    if end_date is None:  # option to bypass Calendar call (which requires a screener connection) for single date
        # data
        dates = [start_date]
    else:
        cal = Calendar.HolidayCal('NYSE')
        dates = sorted(cal.get_range(start_date, end_date).astype(str))
    df_list = []
    for date in dates:
        buffer = io.BytesIO(b'')
        object_name = tickdata.daily_data_object_name(date)
        bucket.downloadToFileObject(object_name, buffer)
        df = pd.read_parquet(buffer)
        if tkrs is not None:
            df = df[df['tkr'].isin(tkrs)].copy()
        df_list.append(df)
    df = pd.concat(df_list)
    df = df.sort_values(['date', 'tkr']).copy()
    df.index = range(len(df))
    return df


def load_minutely_data_summary(start_date, end_date, tkrs):
    """
    tkrs := list of tkrs
    """
    bucket = xgcs.make_conn(tickdata.signed_data_bucket_name())
    if end_date is None:  # option to bypass Calendar call (which requires a screener connection) for single date
        # data
        dates = [start_date]
    else:
        cal = Calendar.HolidayCal('NYSE')
        dates = sorted(cal.get_range(start_date, end_date).astype(str))
    df_list = []
    for date in dates:
        for tkr in tkrs:
            buffer = io.BytesIO(b'')
            object_name = tickdata.minutely_data_object_name(tkr, date)
            objs = bucket.getObjects(prefix=object_name)
            if len(objs) == 1:
                bucket.downloadToFileObject(object_name, buffer)
                df = pd.read_parquet(buffer)
                df_list.append(df)
    df = pd.concat(df_list)
    df = df.sort_values(['date', 'tkr', 'b_ts']).copy()
    df.index = range(len(df))
    return df


def get_flow_summaries(df_in, params=None, metric=None, kind=None, symbol_col=None):
    """
    df_in := should be in the same format as output of prepare_data_summary_for_analysis(), can be a concatenation
        of similar dataframes for multiple days and/or tickers
    metric := usually one of count, qty, notional (notional is the default)
    kind := net, gross, gross_all (this will include unsigned), None (defaults to net)
    symbol_col := by default it uses 'tkr' which comes with signed data, but allows user to switch to a different
        column (if populated)
    """
    if metric is None:
        metric = 'notional'
    if kind is None:
        kind = 'net'
    if symbol_col is None:
        symbol_col = 'tkr'
    if kind not in ['net', 'gross', 'gross_all']:
        raise ValueError('unrecognized value for variable kind')
    df_data = df_in.copy()
    if kind == 'net':
        df_data['value'] = df_data[metric] * df_data['sign']
    elif kind == 'gross':
        df_data['value'] = df_data[metric] * df_data['sign'].abs()
    elif kind == 'gross_all':
        df_data['value'] = df_data[metric].copy()
    else:
        raise ValueError('unrecognized variable kind')
    if params is None:
        params = []
        params.append(dict(tag='all'))
        params.append(dict(tag='dark', venue_type='D'))
        params.append(dict(tag='dark_oddlot', odd_lot=True, venue_type='D'))
        params.append(dict(tag='dark_roundlot', odd_lot=False, venue_type='D'))
        params.append(dict(tag='lit_oddlot', odd_lot=True, venue_type=['L', 'I']))
        params.append(dict(tag='lit_roundlot', odd_lot=False, venue_type=['L', 'I']))
        params.append(dict(tag='iso', iso=True))
        params.append(dict(tag='iso_oddlot', iso=True, odd_lot=True))
        params.append(dict(tag='iso_roundlot', iso=True, odd_lot=False))
        params.append(dict(tag='px_improved', px_improved=True, venue_type='D'))
        params.append(dict(tag='px_improved_oddlot', px_improved=True, odd_lot=True, venue_type='D'))
        params.append(dict(tag='px_improved_roundlot', px_improved=True, odd_lot=False, venue_type='D'))
        params.append(dict(tag='non_iso', iso=False))
        params.append(dict(tag='lit_oddlot_non_iso', odd_lot=True, venue_type=['L', 'I'], iso=False))
        params.append(dict(tag='lit_roundlot_non_iso', odd_lot=False, venue_type=['L', 'I'], iso=False))
    df_list = []
    tag_list = []
    for param in params:
        df = df_data.copy()
        for param_tag, param_value in param.items():
            if param_tag != 'tag':
                if isinstance(param_value, (float, int, str)):
                    df = df[df[param_tag] == param_value].copy()
                else:
                    df = df[df[param_tag].isin(param_value)].copy()
            else:
                tag_list.append(param_value)
        if 'b_ts' in df.columns:
            s_this = df.groupby([symbol_col, 'date', 'b_ts'])['value'].sum()
        else:
            s_this = df.groupby([symbol_col, 'date'])['value'].sum()
        df_list.append(s_this.copy())
    df = pd.concat(df_list, axis=1, keys=tag_list, sort=True).fillna(0).reset_index().copy()
    return df


def get_price(tkr, date, start_time=None, end_time=None, bucket=None, df_signed=None):
    """
    get minutely price from the signed trade dataframe (consider removing after tickdata is incorporated into setock)
    start_time := default is '09:30:00.000000000'
    end_time := default is '16:00'
    note: data is collected for the interval (start_time, end_time]; i.e., start_time is excluded while end_time is
        included
    note: in order to not filter by time at all, one can set parameters, e.g., like this:
        start_time = '', end_time = '24:00'
    bucket := GC bucket where signed trades are; load from GC will be faster when this is passed
    df_signed := can pass signed data in df_signed instead of loading them from the raw signed data file; if
        df_signed is not None, bucket parameter is ignored
    """
    if start_time is None:
        start_time = '09:30:00.000000000'
    if end_time is None:
        end_time = '16:00'
    # raw dataframe
    if df_signed is None:
        if date == datetime.strftime(datetime.today(), '%Y-%m-%d'):
            df_raw = get_signed_trades_live(tkr, date)
        else:
            df_raw = tickdata.get_signed_trades(tkr, date, bucket=bucket)
    else:
        df_raw = df_signed.copy()
    # minutely tags
    ts_tags = sorted(['{0:02d}:{1:02d}'.format(h, m) for h in range(24) for m in range(60)])
    ts_breakpoints = [int(v[:2]) * 3600 + int(v[3:5]) * 60 for v in ts_tags]
    ts_tags = ts_tags[1:]
    # minutely dataframe
    if date < '2015-09-01':
        time_col = 'time'
    else:
        time_col = 'e_time'
    df = df_raw.copy()
    df = df[df['exclude'] != 'X'].copy()
    df = df[(df[time_col] > start_time) & (df[time_col] <= end_time)].copy()
    df['ts'] = [int(v[:2]) * 3600 + int(v[3:5]) * 60 + float(v[6:]) for v in df[time_col]]
    df['b_ts'] = pd.cut(df['ts'], ts_breakpoints, labels=ts_tags).astype(str)
    df = df.sort_values(['b_ts', time_col]).copy()
    s_px = df.groupby(['b_ts'])['px'].last()
    return s_px


def get_signed_trades_live(tkr=None, date: str = None,
                           start_dt=None, end_dt=None,
                           tkr_list=[], dt_to_str=True, optimization='arrow'):
    """
    Fetches live signed trades for a given ticker and date.
    If date is specified, the function will return the data for the date, and start_dt and end_dt will be ignored
    Note:
        - this function requires access to KDB sandbox
        - BRK/A and BRK/B are mapped to BRKA and BRKB in the output dataframe to align with TickData format.

    Parameters:
    tkr (str): The ticker symbol for which to fetch signed trades.
    date (str): The date for which to fetch signed trades in 'YYYY-MM-DD' format.
    start_dt (pd.Timestamp): The start date for which to fetch signed trades.
    end_dt (pd.Timestamp): The end date for which to fetch signed trades.
    tkr_list (list): The list of ticker symbols for which to fetch signed trades.
    dt_to_str (bool): Whether to convert the date and time columns to strings.

    Returns: pd.DataFrame
    """
    ticker_vela_mapper = {
        'BRKA':'BRK/A',
        'BRKB':'BRK/B'
    }  # map to vela ticker format

    # format input parameters
    if tkr is not None and len(tkr_list) > 0:
        raise ValueError('tkr and tkr_list cannot be specified at the same time')
    if tkr is not None:
        tkr_list = [tkr]
    if date is not None:
        start_dt = pd.Timestamp(date + " 01:00:00")
        end_dt = pd.Timestamp(date + " 23:59:59")
    else:
        if start_dt is None:
            start_dt = pd.Timestamp(datetime.today().strftime('%Y-%m-%d') + " 01:00:00")
        if end_dt is None:
            end_dt = pd.Timestamp(datetime.today().strftime('%Y-%m-%d') + " 23:59:59")
        date = start_dt.strftime('%Y-%m-%d')

    # Convert tkr_list values if they are in ticker_vela_mapper
    tkr_list = [ticker_vela_mapper.get(tkr, tkr) for tkr in tkr_list]

    # get nbbo trades
    df1 = kdb.getHistSignedTrade(tkr_list, start_dt, end_dt, d=optimization)
    df1['tkr'] = df1['sym']
    df1['sym'] = df1['sym'] + '.' + df1['exch']
    # get all exchange trades
    tkr_ex_list = df1['sym'].unique().tolist()
    df2 = kdb.getHistSignedTrade(tkr_ex_list, start_dt, end_dt, d=optimization)  # result: doesn't include dark trades as X.D ticker
    # combine dark trades + exchange trades
    df_signed = pd.concat([df1.loc[df1["exch"] == "D"], df2], axis=0)

    # remove error rows that have exchange time older than the requested date
    df_signed = df_signed.loc[
        df_signed["exchTime"] > date
    ]

    df_signed['tkr'] = df_signed['sym'].str.split('.').str[0]
    vela_ticker_mapper = {v: k for k, v in ticker_vela_mapper.items()}
    df_signed['tkr'] = df_signed['tkr'].map(lambda tkr: vela_ticker_mapper.get(tkr, tkr))

    # add exclude annotation, refer to TickData documentation :
    # No correction analogous available in EOD
    exclude_cond_code_list = ['B','C', 'G', 'L', 'M', 'N', 'P', 'Q', 'R', 'W', 'Z', '8', '9']
    df_signed['exclude'] = df_signed['cond'].map(lambda cond: 'X' if any(c in exclude_cond_code_list for c in cond) else None)

    # exclude I during market hours, the real market time is not going to match precisely
    df_signed['time'] = pd.to_datetime(df_signed['time'], format='%H:%M:%S.%f').dt.time
    df_signed.loc[(df_signed['time'] >= MKT_OPEN_TIME) & (df_signed['time'] <= MKT_CLOSE_TIME) & (df_signed['cond'].str.contains('I')), 'exclude'] = 'X'

    # add trade through exempt indication, refer to TickData documentation
    trade_through_exempt_cond_code_list = ['F','Q','3','5', '6']
    df_signed['trade_through_exempt'] = df_signed['cond'].map(lambda cond: 1 if any(c in trade_through_exempt_cond_code_list for c in cond) else 0)

    # check for duplicates in df_signed by id, sym, seqNum
    duplicate_rows = df_signed[df_signed.duplicated(subset=['id', 'sym', 'seqNum'], keep=False)]
    if not duplicate_rows.empty:
        print(f"Duplicate rows in df_signed based on id, sym, seqNum: {duplicate_rows}")

    # add nbbo columns
    df_signed = df_signed.merge(df1[['id','sym','seqNum', 'sign','bidPrice','askPrice']]\
        , on=['id','sym','seqNum'], how='left', suffixes=('_venue','_nbbo'))

    if dt_to_str:
        # time consuming operation
        df_signed["date"] = df_signed["exchTime"].dt.strftime("%m/%d/%Y")
        df_signed["time"] = df_signed["srcTime"].dt.strftime("%H:%M:%S.%f")
        df_signed["exchTime"] = df_signed["exchTime"].dt.strftime("%H:%M:%S.%f")
    else:
        df_signed["time"] = df_signed["srcTime"]  # use srcTime from data vendors
    df_signed.rename(
        columns={
            "price": "px",
            "size": "qty",
            "exch": "venue",
            "cond": "condition",
            "seqNum": "seq",
            "exchTime": "e_time",
            "bidPrice_venue": "venue_bid",
            "askPrice_venue": "venue_ask",
            "sign_venue": "venue_sign",
            "bidPrice_nbbo": "nbbo_bid",
            "askPrice_nbbo": "nbbo_ask",
            "sign_nbbo": "nbbo_sign",
        },
        inplace=True,
    )

    df_signed = df_signed.reindex(
        columns=[
            "date",
            "time",
            'tkr',
            "px",
            "qty",
            "venue",
            "condition",
            "correction",
            "seq",
            "trade_stop",
            "source",
            "trf",
            "exclude",
            "filtered_px",
            "e_time",
            "id",
            "trf_time",
            "time_obsolete",
            "participant_time_obsolete",
            "trf_time_obsolete",
            "trade_through_exempt",
            "nbbo_bid",
            "nbbo_ask",
            "nbbo_sign",
            "venue_bid",
            "venue_ask",
            "venue_sign",
        ]
    )
    return df_signed


def is_dst(date:datetime, timezone='US/Eastern'):
    """
    check date is DST in the given timezone
    """
    eastern = pytz.timezone(timezone)
    date = eastern.localize(date)
    return bool(date.dst())


def get_quotes(tkr, dt, start_time='09:30:00', end_time='16:05:00', use_pit=False):
    column_names = ['date', 'time', 'venue', 'bid', 'ask', 'qty_bid', 'qty_ask', 'seq_num','e_time']
    if dt == pd.Timestamp('today').strftime('%Y-%m-%d') or use_pit == True:
        # Get tickdata as we received from PI feed
        symbols = [tkr + '.' + chr(i) for i in range(ord('A'), ord('Z')+1)]
        print(f'Loading quotes from Real Time KDB for {tkr} on {dt}')
        df = kdb.getHistQuote(symbols, dt + ' ' + start_time, dt + ' ' + end_time)
        if not (df['bidExch'] == df['askExch']).all():
            raise ValueError('Single Venue cannot have different bid/ask exchanges')
        original_columns = ['date', 'srcTime', 'bidExch', 'bidPrice', 'askPrice', 'bidSize', 'askSize', 'seqNum','exchTime']
        df = df[original_columns]
        df.columns = column_names
        df['time'] = pd.to_datetime(df['time']).dt.time
        df['e_time'] = pd.to_datetime(df['e_time']).dt.time
        df['qty_bid'] = df['qty_bid'] / 100  # convert to 100 share units
        df['qty_ask'] = df['qty_ask'] / 100  # convert to 100 share units
    else:
        # Get tickdata from EOD file
        print(f'Loading quotes from EOD for {tkr} on {dt}')
        column_indices = [0, 1, 2, 3, 4, 5, 6, 23, 9]
        column_map = {idx: name for idx, name in zip(column_indices, column_names)}
        df = tickdata.load_quotes(tkr, dt, include_sizes=True, column_map=column_map)
        # Filter df based on 'time'
        df['temp'] = pd.to_timedelta(df['time'])
        start_time_td = pd.to_timedelta(start_time)
        end_time_td = pd.to_timedelta(end_time)
        df = df[(df['temp'] >= start_time_td) & (df['temp'] <= end_time_td)]
        df = df.drop(columns=['temp']).reset_index(drop=True)
    return df


def get_signed_trades(tkr, dt, keep_original_names=False, optimization='arrow'):
    if dt == pd.Timestamp('today').strftime('%Y-%m-%d'):
        # print(f'Loading signed trades from Real Time KDB for {tkr} on {dt}')
        df=get_signed_trades_live(tkr,dt, optimization=optimization)
        if keep_original_names:
            df = df.rename(columns=tickdata.original_column_name_map())
    else:
        # print(f'Loading signed trades from EOD for {tkr} on {dt}')
        from xlib import xbq, xcredfile
        xcredfile.set_gcp_credentials()
        df=tickdata.get_signed_trades(tkr, dt, keep_original_names=keep_original_names)
    return df


def flow_summary_for_retail(date, tkrs):
    """
    :param date: YYYY-MM-DD
    :param tkrs: list of tickers
    :return: returns a dict with ISO and EDGX flows
    """
    df_map = gd.getData('sstock-identifiers', mappings=['gvid:ticker'], date='yest', countries=['US'], tickers=tkrs)
    s_gvid_to_tkr = pd.Series(list(df_map.iloc[0]), df_map.columns)
    gvids = list(s_gvid_to_tkr.index)
    map_this = {'RetailFlow_Minute_Stock_Signed_Trades.K_net_dol': 'EDGX',
                'RetailFlow_Minute_Stock_Signed_Trades.iso_net_dol': 'ISO'}
    data = gd.getData('sstock-v2', universe='AllStkNA', date=date, fields=sorted(map_this.keys()), symbols=gvids)
    out = dict()
    for field_name, df_values in data.items():
        df = df_values.multiply(1e6).copy()
        df = df.tz_localize(None).copy()
        df.columns = list(s_gvid_to_tkr.loc[df.columns])
        out[map_this[field_name]] = df.copy()
    return out
