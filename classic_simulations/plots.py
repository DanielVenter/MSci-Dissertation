import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default='svg'


def plot_histogram(pot_sizes: list, 
                   upper_confidence:float, 
                   lower_confidence: float):
    """
    Plots the frequencies of the final pot sizes.
    """
    fig = px.histogram(pot_sizes, 
                       title=f"The final pot size after {len(pot_sizes)} simulations.")
    
    fig.add_vline(x=lower_confidence, 
                  line_width=3, 
                  line_dash="dash", 
                  line_color="green")
    
    fig.add_vline(x=upper_confidence, 
                  line_width=3, 
                  line_dash="dash", 
                  line_color="green")
    
    fig.add_vline(x=np.median(pot_sizes), 
                  line_width=3, 
                  line_dash="dash", 
                  line_color="black",
                  annotation_text="median",
                  annotation_font_size=15)
    
    fig.add_vrect(x0=lower_confidence, 
                  x1=upper_confidence, 
                  line_width=0, 
                  fillcolor="green",
                  opacity=0.2,
                  annotation_text="95% confidence interval",
                  annotation_font_size=15)
    
    fig.update_layout(
        xaxis_title="Amount (£)",
        yaxis_title="Count",
        showlegend=False,
        font=dict(
            family="Arial",
            size=14
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    fig.write_html('outputs/mc-histogram.html', auto_open=False)


def plot_yearly_percentiles(inputs, df):
    """
    Plots the year by year percentile graph.
    """
    exact_np = df[df['90th_percentile'] > inputs['target_amount']].iloc[0]
    exact_sfp = df[df['75th_percentile'] > inputs['target_amount']].iloc[0]
    exact_median = df[df['median'] > inputs['target_amount']].iloc[0]
    exact_tfp = df[df['25th_percentile'] > inputs['target_amount']].iloc[0]
    exact_tp = df[df['10th_percentile'] > inputs['target_amount']].iloc[0]
    
    fig = go.Figure()
    
    fig.add_traces(go.Scatter(x=df['age'], 
                              y=df['10th_percentile'],
                              line = dict(color='#FFA502'),
                              mode='lines',
                              name='10th %tile',
                              fill='none', 
                              fillcolor = '#F7CA77'))
    
    fig.add_traces(go.Scatter(x=df['age'], 
                              y=df['25th_percentile'],
                              line = dict(color='#7BE56E'),
                              mode='lines',
                              name='25th %tile',
                              fill='tonexty', 
                              fillcolor = '#F7CA77'))

    
    fig.add_traces(go.Scatter(x=df['age'], 
                              y=df['median'],
                              line=dict(color='black'),
                              line_width=3,
                              mode='lines',
                              name="median", 
                              fill='tonexty',
                              fillcolor='#00FF66'))
    
    fig.add_traces(go.Scatter(x=df['age'], 
                              y=df['75th_percentile'],
                              line = dict(color='#7BE56E'),
                              mode='lines',
                              name="75th %tile",
                              fill='tonexty', 
                              fillcolor = '#00FF66'))
    
    fig.add_traces(go.Scatter(x=df['age'], 
                              y=df['90th_percentile'],
                              line = dict(color='#FFA502'),
                              mode='lines',
                              name="90th %tile",
                              fill='tonexty', 
                              fillcolor = '#F7CA77'))
    
    fig.update_layout(hovermode="x")
    
    fig.update_xaxes(tickangle=0, 
                     dtick=1,
                     showticklabels=True, 
                     gridcolor='lightgray',
                     type='category')
    
    fig.update_yaxes(gridcolor='lightgray',
                     rangemode="tozero")
    
    fig.add_hline(y=inputs['target_amount'], 
                  line_width=2, 
                  line_dash='dash', 
                  line_color='red',
                  annotation_text='Target amount',
                  annotation_font=dict(
                    family="Arial",
                    size=15,
                    color="red"
                  ),
                  annotation_font_size=15,
                  annotation_position='bottom left',
                  fillcolor='red')
    
    fig.add_shape(type="line",
                  x0=int(exact_median['year'] - 1), 
                  y0=0, 
                  x1=int(exact_median['year'] - 1), 
                  y1=inputs['target_amount'],
                  line_width=2,
                  line_color='gray',
                  line_dash='dash')
    
    fig.add_shape(type="line",
                  x0=int(exact_tp['year'] - 1), 
                  y0=0, 
                  x1=int(exact_tp['year'] - 1), 
                  y1=inputs['target_amount'],
                  line_width=2,
                  line_color='orange',
                  line_dash='dash')
    
    fig.add_shape(type="line",
                  x0=int(exact_np['year'] - 1), 
                  y0=0, 
                  x1=int(exact_np['year'] - 1), 
                  y1=inputs['target_amount'],
                  line_width=2,
                  line_color='orange',
                  line_dash='dash')
    
    fig.add_shape(type="line",
                  x0=int(exact_tfp['year'] - 1), 
                  y0=0, 
                  x1=int(exact_tfp['year'] - 1), 
                  y1=inputs['target_amount'],
                  line_width=2,
                  line_color='green',
                  line_dash='dash')
    
    fig.add_shape(type="line",
                  x0=int(exact_sfp['year'] - 1), 
                  y0=0, 
                  x1=int(exact_sfp['year'] - 1), 
                  y1=inputs['target_amount'],
                  line_width=2,
                  line_color='green',
                  line_dash='dash')
      
    fig.add_annotation(x=int(exact_median['year'] - 1), 
                       y=inputs['target_amount'] * 1.45,
                       text=f"<b>{int(exact_median['year'])} years</b>",
                       font=dict(
                            color="black",
                            size=21
                       ),
                       showarrow=False,
                       yshift=10)
    
    fig.add_annotation(x=int(exact_median['year'] - 1), 
                       y=inputs['target_amount'] * 1.3,
                       text=f"<b>(Age {int(exact_median['age'])})</b>",
                       font=dict(
                            color="black",
                            size=21
                       ),
                       showarrow=False,
                       yshift=10)
    
    fig.add_annotation(x=inputs['end_age'] - inputs['start_age'] - 1.2, 
                       y=df['10th_percentile'].max() - 8000,
                       text="<b>10%</b>",
                       font=dict(
                            color="black",
                            size=12
                       ),
                       showarrow=False,
                       yshift=10)
    
    fig.add_annotation(x=inputs['end_age'] - inputs['start_age'] - 1.2, 
                       y=df['25th_percentile'].max() - 8000,
                       text="<b>25%</b>",
                       font=dict(
                            color="black",
                            size=12
                       ),
                       showarrow=False,
                       yshift=10)
        
    fig.add_annotation(x=inputs['end_age'] - inputs['start_age'] - 1.25, 
                       y=df['median'].max() - 5000,
                       text="<b>median</b>",
                       font=dict(
                            color="black",
                            size=12
                       ),
                       showarrow=False,
                       yshift=10)
    
    fig.add_annotation(x=inputs['end_age'] - inputs['start_age'] - 1.2, 
                       y=df['75th_percentile'].max() - 4000,
                       text="<b>75%</b>",
                       font=dict(
                            color="black",
                            size=12
                       ),
                       showarrow=False,
                       yshift=10)
    
    fig.add_annotation(x=inputs['end_age'] - inputs['start_age'] - 1.2, 
                       y=df['90th_percentile'].max() - 5000,
                       text="<b>90%</b>",
                       font=dict(
                            color="black",
                            size=12
                       ),
                       showarrow=False,
                       yshift=10)
    
    fig.add_annotation(x=.99,
                       xref='paper',
                       xanchor='right',
                       y=0,
                       yanchor='bottom',
                       text="<b>shedloadofcode.com</b>",
                       font=dict(
                            color="gray",
                            size=14
                       ),
                       showarrow=False)
    
    fig.add_annotation(x=0.01,
                       xref='paper',
                       yref='paper',
                       xanchor='left',
                       y=0.99,
                       yanchor='top',
                       text=f"In <b>{inputs['n_simulations']}</b> simulations " +
                            f"<b>{int(exact_median['age'])}</b> " +
                            f"is the median age ({int(exact_median['year'])} years)<br>",
                       font=dict(
                            color="black",
                            size=15
                       ),
                       showarrow=False)
    
    fig.add_annotation(x=0.01,
                       xref='paper',
                       yref='paper',
                       xanchor='left',
                       y=0.96,
                       yanchor='top',
                       text="<span style=\"color:orange\">10th to 90th %ile: " +
                            f"<b>{int(exact_np['year'])} to {int(exact_tp['year'])} " + 
                             "years</b> to target</span>",  
                       font=dict(
                            color="black",
                            size=15
                       ),
                       showarrow=False)
    
    fig.add_annotation(x=0.01,
                       xref='paper',
                       yref='paper',
                       xanchor='left',
                       y=0.93,
                       yanchor='top',
                       text="<span style=\"color:green\">25th to 75th %ile: " +
                            f"<b>{int(exact_sfp['year'])} to {int(exact_tfp['year'])} " + 
                             "years</b> to target</span>",
                       font=dict(
                            color="black",
                            size=15
                       ),
                       showarrow=False)
     
    
    fig.update_layout(
        title=f"Percentiles by year after {inputs['n_simulations']} simulations.",
        xaxis_title="Age",
        yaxis_title="Amount (£)",
        legend_title="Legend Title",
        showlegend=False,
        font=dict(
            family="Arial",
            size=14
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    fig.write_html('outputs/mc-percentiles.html', auto_open=False)
