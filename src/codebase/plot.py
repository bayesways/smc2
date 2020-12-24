import pandas as pd
import numpy as np
import altair as alt

def form_df(samples, num_chains, rows):
    dfs = []
    for cn in range(num_chains):
        for r in range(rows):
            if rows>1:
                df = pd.DataFrame(
                        samples[:,cn, r]
                )
            else:
                df = pd.DataFrame(
                        samples[:,cn]
                )
            df.insert(0, 'idx', np.arange(df.shape[0]))
            df = df.melt(id_vars ='idx', var_name = 'col')
            df.insert(1, 'row' , r)
            df.insert(1, 'chain', cn)
            dfs.append(df)
    return pd.concat(dfs).reset_index(drop=True)


def get_post_df(samples):
    num_chains = samples.shape[1]
    if samples.ndim > 3:
        rows = samples.shape[2]
        df = form_df(samples, num_chains, rows)
    else:
        rows = 1
        df = form_df(samples, num_chains, 1)
    return df


def plot_density(df, width=300, height=50) :
    # it only works for one chain for now
    assert df.chain.unique().shape[0] == 1
    return alt.Chart(df).transform_fold(
        ['value']
    ).transform_density(
        density='value',
        groupby=['cn', 'row', 'col', 'source'],
    ).mark_area(opacity=0.5).encode(
        alt.X('value:Q', title=None),
        alt.Y('density:Q'),
        alt.Row('row'),
        alt.Column('col'),
        alt.Color('source')
    ).resolve_scale(
        x='independent'
    ).properties(width=width, height=height)


def plot_line(df, width=300, height=50) :
    # it only works for one chain for now
    assert df.chain.unique().shape[0] == 1
    return alt.Chart(df).mark_line(
        strokeWidth = 1,
        ).encode(
        alt.X('idx:Q', title=None),
        alt.Y('value:Q'),
        alt.Row('row'),
        alt.Column('col'),
        alt.Color('source')
    ).resolve_scale(
        x='independent'
    ).properties(width=width, height=height)

