import pandas as pd
import numpy as np
import xlib.xdb as xdb
import vol_generic.dbutil as dbutil
import VolUtilities.api as vu
import gzip

def calc_moments(df_in, m=4):
    p = df_in['cdf'].diff()
    p = p.fillna(0)/2 + p.shift(-1).fillna(0)/2
    p.iloc[0] += df_in['cdf'].iloc[0]
    p.iloc[-1] += 1-df_in['cdf'].iloc[-1]
    df_in['p'] = p
    mo = pd.Series(index=range(1,m+1), dtype=float)
    for n in mo.index:
        if n <=1:
            df_in['f'] = df_in['x']**n
        else:
            df_in['f'] = (df_in['x']-mo[1])**n
        mo[n] = (df_in['f']*df_in['p']).sum()
    return mo

def load_option_spiderrock(date=None):
    max_dte = 1
    if date is None:
        date = dbutil.calUS().get_date().strftime("%Y-%m-%d")
    ticker_list = ['SPX']
    maxExpiry = dbutil.calUS().get_nth_date(date, max_dte).strftime('%F')

    sql = f"""
    select importdt,okey_tk,okey_yr,okey_mn,okey_dy,okey_xx,okey_cp,ticker_tk,
    uprc,years,rate,obid,oask, de,timestamp,timestamp_us
    from [SpiderRockTrading.V7].[dbo].msgoptionimpliedquoteadj
    where 1=1
    """
    if len(ticker_list) > 0:
        sql += f"""and ticker_tk in ('{"','".join(ticker_list)}')"""

    sql += f"""
    and CONVERT(DATE, CONCAT(
    RIGHT('0000' + CAST(okey_yr AS VARCHAR(4)), 4), '-',
    RIGHT('00' + CAST(okey_mn AS VARCHAR(2)), 2), '-',
    RIGHT('00' + CAST(okey_dy AS VARCHAR(2)), 2)
    )
    ) <= '{maxExpiry}' """
    if date:
        sql += f"""\nand importdt >= '{date}' """
        sql += f"""\nand importdt <= '{date} 23:59:59'"""

    sql += """
    order by importdt, timestamp, timestamp_us"""
    # print(sql)
    db = xdb.make_conn('SPIDERROCKTRADING.V7')
    df_sr = db.query(sql)

    df_sr["expiry"] = pd.to_datetime(
        df_sr["okey_yr"].astype(str).str.zfill(4)
        + df_sr["okey_mn"].astype(str).str.zfill(2)
        + df_sr["okey_dy"].astype(str).str.zfill(2)
    ).dt.strftime("%Y-%m-%d")
    df_sr.drop(['okey_yr', 'okey_mn', 'okey_dy'], axis=1, inplace=True)

    df_sr.rename(columns={'okey_tk': 'cls', 'ticker_tk': 'tkr', 'okey_cp': 'put_call', 'okey_xx': 'strike'}, inplace=True)
    df_sr['put_call'] = df_sr['put_call'].str[:1]
    df_sr['mid'] = (df_sr['obid'] + df_sr['oask'])/2

    df_sr['isAM'] = df_sr['tkr'].isin(['SPX', 'NDX', 'RUT'])&(df_sr['tkr']==df_sr['cls'])
    df_sr = df_sr[~df_sr['isAM']]
    return df_sr

def calc_expiry_tm(opts, expiry, date, tm, maxSig):
    """
    returns df_pt, res
    """
    expiryNS = expiry.replace('-', '')
    # opts = opts[opts['oask']>0]
    if opts.empty: return None, None
    # find forward and discount factor
    df_x = opts.pivot_table(index=['strike'], columns=['put_call'], values='price').dropna()
    if df_x.shape[1] < 2: return None, None
    df_x['x'] = df_x['C'] - df_x['P']
    df_x['strike'] = df_x.index
    atmStrikes = df_x['x'].abs().nsmallest(4).index.values

    k0 = atmStrikes[:2].mean()
    kmin = df_x.index[0]
    kmax = df_x.index[-1]
    kr = 0.5
    k1t = kmin * kr + k0 * (1 - kr)
    k2t = kmax * kr + k0 * (1 - kr)
    k1, x1 = df_x.loc[(df_x['strike'] - k1t).abs().idxmin(), ['strike', 'x']]
    k2, x2 = df_x.loc[(df_x['strike'] - k2t).abs().idxmin(), ['strike', 'x']]
    # if k1 >= k2: raise ValueError('Invalid expiry strip')
    if k1 >= k2: return None, None
    df = -(x1 - x2) / (k1 - k2)

    atmStrikes = df_x['x'].abs().nsmallest(4).index
    forw = (df_x.loc[atmStrikes, 'x'] / df + atmStrikes).mean()
    # opts = opts[opts['bid']>0.0]

    df_con = []
    for pc in ['C', 'P']:
        price = opts[opts['put_call']==pc].groupby(['strike'])['price'].mean()
        if pc=='P': price = price.iloc[::-1]
        price = price[price == price.cummin()]
        price = price[price.diff().fillna(-1)<0]
        df_c = price.to_frame('price')
        df_c['strike'] = df_c.index
        df_c['cdf'] = df_c['price'].diff()/df_c['strike'].diff()
        df_c['cdf2'] = pd.concat([df_c['cdf'], df_c['cdf'].shift(-1)], axis=1).mean(axis=1)
        df_con.append(df_c['cdf2'].to_frame('cdf'+pc))
    df_con = pd.concat(df_con, axis=1).sort_index()
    df_pt = df_con.copy()
    df_pt['strike'] = df_pt.index

    k0 = forw
    df_pt['df'] = df
    df_pt['forw'] = forw
    df_pt['cdfC'] = df_pt['cdfC'] / df_pt['df'] + 1
    df_pt['cdfP'] /= df_pt['df']
    df_pt['cdf'] = df_pt['cdfP'].where(df_pt['strike'] < k0, df_pt['cdfC'])

    #new in v2, skips nans
    df_pt.loc[k0:, 'cdf'] = df_pt.loc[k0:, 'cdf'].cummax().clip(upper=1)
    df_pt.loc[:k0, 'cdf'] = df_pt.loc[:k0, 'cdf'].iloc[::-1].cummin().clip(lower=0).iloc[::-1]

    df_pt = df_pt.dropna(subset=['cdf'])
    #remove redundant strikes
    kMin = df_pt[df_pt['cdf']==df_pt['cdf'].iloc[0]].index[-1]
    kMax = df_pt[df_pt['cdf']==df_pt['cdf'].iloc[-1]].index[0]
    df_pt = df_pt.loc[kMin:kMax]

    df_pt['x'] = df_pt['strike'] / forw - 1
    tte = vu.getTimeToExpiry(pd.to_datetime(date+' '+tm), expiryNS)[0]

    mo = calc_moments(df_pt)
    df_pt = df_pt[np.abs(df_pt['x']) < maxSig*np.sqrt(mo[2])]
    mo = calc_moments(df_pt)
    res = {'date':date, 'tm':tm, 'expiry': expiry}
    res['sigma'] = np.sqrt(mo[2])
    res['skew'] = mo[3]/res['sigma']**3
    res['kurt'] = mo[4]/res['sigma']**4
    res['forw'] = forw
    res['df'] = df
    res['tte'] = tte
#    res['m0Err'] = 1 - mo[0]
    res['m1Err'] = mo[1]
    return df_pt, res

def calc_intra_im(date, ric='.SPX', maxSig=4):
    dateInt = int(date.replace('-', ''))
    fpath = '/data/vol-intraday/price2/ric_%s/%d/defs.csv.gz'%(ric, dateInt)
    f = gzip.open(fpath, 'rb')
    df_defs = pd.read_csv(f)
    fpath = '/data/vol-intraday/price2/ric_%s/%d/bar.csv.gz'%(ric, dateInt)
    f = gzip.open(fpath, 'rb')
    optData = pd.read_csv(f)

    mask = (df_defs['put_call']!='S')&(df_defs['weekly'])
    df_exp = df_defs[mask].groupby(['expiry', 'weekly']).size().to_frame('opt_cnt').reset_index()
    df_exp['dte'] = df_exp['expiry'].map(lambda x: dbutil.calUS().diff(date, x))

    df_exp = df_exp[df_exp['dte']<=1]
    tmList = np.array([str(t.time())[:-3] for t in pd.date_range(start='09:30', end='16:00', freq='1min')])
    # tmList = ['09:33']

    df_res = []
    for e, erow in df_exp.iloc[:].iterrows():
        expiry = erow['expiry']
        df_sel = df_defs[(df_defs['expiry']==expiry)&(df_defs['weekly']==erow['weekly'])]
        optSub = optData[['symId', 'mid', 'bid', 'ask', 'tm']].join(
            df_sel.set_index('symId')[['strike', 'put_call']], on=['symId'], how='inner')
        optSub.rename(columns={'mid': 'price'}, inplace=True)

        for tm in tmList[:]:
            opts = optSub[optSub['tm']==tm]
            df_pt, res = calc_expiry_tm(opts, expiry, date, tm, maxSig)
            if res is None:
                continue
            df_res.append(res)

    if len(df_res) == 0: return None
    df_res = pd.DataFrame(df_res)
    return df_res

def calc_im_live(date=None, maxSig=4, return_cdf=False):
    if date is None:
        date = dbutil.calUS().get_date().strftime("%Y-%m-%d")
    optData = load_option_spiderrock(date)
    optData['timestamp'] = optData['timestamp'].dt.tz_localize('America/Chicago').dt.tz_convert('America/New_York').dt.tz_localize(None)
    optData['dt'] = optData['timestamp'].dt.ceil('1T')
    optData['tm'] = optData['dt'].astype(str).str[11:16]
    mask = optData['oask']>0
    mask &= ~((optData['obid']==0)&(optData['oask']>50))
    optData = optData[mask]

    cnt = optData.groupby(['tm']).size()
    tmList = cnt.index.values[:]
    expiryList = optData['expiry'].unique()

    df_res = []
    cdf_dict = dict()
    for expiry in expiryList:
        optSub = optData[optData['expiry']==expiry].copy()
        optSub.rename(columns={'mid': 'price'}, inplace=True)

        if return_cdf:
            cdf_dict[expiry] = dict()

        for tm in tmList[:]:
            opts = optSub[optSub['tm']==tm]
            df_pt, res = calc_expiry_tm(opts, expiry, date, tm, maxSig)
            if df_pt is None:
                continue
            df_res.append(res)
            if return_cdf:
                cdf_dict[expiry][tm] = df_pt

    if len(df_res) == 0: return None
    df_res = pd.DataFrame(df_res)

    if return_cdf:
        return df_res, cdf_dict
    else:
        return df_res
