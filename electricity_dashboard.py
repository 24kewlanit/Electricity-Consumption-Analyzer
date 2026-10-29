from pathlib import Path
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, streamlit as st, plotly.express as px, plotly.graph_objects as go
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Electricity Consumption Analyzer", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")
NAVY, BLUE, AMBER, STEEL = "#14213D", "#1F3A5F", "#C98A12", "#3D5A80"
COLORS = [BLUE, AMBER, STEEL, "#7A8CA5", "#2E7D6B", "#8C4A4A"]
TARGET = "Global_active_power"
FEATURES = ["Global_reactive_power", "Voltage", "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
REQ = [TARGET] + FEATURES
st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:wght@400;700&family=Inter:wght@400;500;600;700&display=swap');
.stApp{background:#F4F5F9;color:#202B3B;font-family:Inter,Arial,sans-serif}
[data-testid="stSidebar"]{background:#14233F;min-width:205px}
[data-testid="stSidebar"] *{color:#F5F7FB!important}
[data-testid="stSidebar"] [data-testid="stRadio"] label{padding:8px 10px;border-radius:4px;font-size:13px;min-height:34px;cursor:pointer;border-left:3px solid transparent}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked){background:#243A5B;border-left-color:#D49A22}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover{background:#243A5B}
[data-testid="stWidgetLabel"] p{font-size:12px!important;font-weight:600!important;color:#34445B!important}
[data-testid="stSelectbox"] label,[data-testid="stNumberInput"] label,[data-testid="stSlider"] label,[data-testid="stRadio"] label{visibility:visible!important;opacity:1!important}
[data-baseweb="select"]>div{background:#fff!important;border-color:#CFD7E3!important}
[data-testid="stNumberInput"] input{background:#fff!important;color:#172944!important}
[data-testid="stFormSubmitButton"] button{min-height:40px;font-size:13px}
[data-testid="stPlotlyChart"]{border:1px solid #E0E5ED;border-radius:5px;padding:3px;background:#fff}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover{background:#243A5B}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p{font-size:11px;color:#C4D0E0!important}
.main .block-container{max-width:1200px;padding-top:1.6rem;padding-bottom:1.2rem}
h1,h2,h3,h4{font-family:'Libre Baskerville',Georgia,serif;color:#172944!important}
h1{font-size:27px!important} h2{font-size:22px!important} h3{font-size:17px!important} h4{font-size:14px!important}
p,li,label,div[data-testid="stCaptionContainer"]{font-size:12px}
div[data-testid="stMetric"]{background:#fff;border:1px solid #DFE4EC;border-left:3px solid #203A60;border-radius:5px;padding:13px 14px;box-shadow:none}
div[data-testid="stMetricLabel"]{font-size:11px;color:#687385}
div[data-testid="stMetricValue"]{font-family:'Libre Baskerville',Georgia,serif;color:#1B2D4B;font-size:25px}
.stButton>button, .stFormSubmitButton>button{background:#203A60;color:#fff;border:0;border-radius:5px;font-weight:600;padding:8px 18px}
.stButton>button:hover,.stFormSubmitButton>button:hover{background:#C28A17;color:#fff}
div[data-testid="stDataFrame"]{border:1px solid #DFE4EC;border-radius:5px;overflow:hidden}
section[data-testid="stSidebar"] hr{border-color:#31425D}
hr{border-color:#E0E5ED}
div[data-testid="stAlert"]{border-radius:5px}
.brand{font-family:'Libre Baskerville',Georgia,serif;font-size:14px;font-weight:700;line-height:1.4;padding:8px 0 18px;border-bottom:1px solid #30415C;margin-bottom:15px;color:#F8E5B0}
.brand small{display:block;font-family:Inter,Arial,sans-serif;font-size:9px;font-weight:400;color:#B9C7D9;margin-top:5px}
.note{background:#E2EEFF;border-left:3px solid #5279AC;padding:10px 12px;border-radius:4px;font-size:11px;color:#24466F}
.footer{border-top:1px solid #E0E5ED;margin-top:26px;padding-top:12px;text-align:center;color:#778195;font-size:10px}
</style>''', unsafe_allow_html=True)
st.sidebar.markdown('<div class="brand">⚡ Electricity<br>Consumption Analyzer<small>Household power analytics</small></div>', unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def clean_data(raw, filename):
    f=BytesIO(raw)
    try: d=pd.read_csv(f,sep=';',low_memory=False)
    except: f.seek(0); d=pd.read_csv(f,low_memory=False)
    if len(d.columns)<3 or TARGET not in d.columns:
        f.seek(0); d=pd.read_csv(f,low_memory=False)
    d.columns=[str(c).strip() for c in d.columns]
    if TARGET not in d.columns: raise ValueError('Wrong file format or missing Global_active_power column.')
    missing=[c for c in REQ if c not in d.columns]
    if missing: raise ValueError('Missing columns: '+', '.join(missing))
    for c in REQ: d[c]=pd.to_numeric(d[c].replace('?',np.nan),errors='coerce')
    d=d.dropna(subset=REQ).copy()
    if d.empty: raise ValueError('No complete numeric rows after cleaning.')
    if 'Date' in d.columns and 'Time' in d.columns:
        dt=pd.to_datetime(d['Date'].astype(str)+' '+d['Time'].astype(str),dayfirst=True,errors='coerce')
        d['DateTime']=dt; d['Hour']=dt.dt.hour; d['Weekday']=dt.dt.day_name(); d['Month']=dt.dt.to_period('M').astype(str)
    else:
        for c in ['DateTime','Hour','Weekday','Month']: d[c]=np.nan
    median=float(d[TARGET].median()); d['Consumption_Level']=np.where(d[TARGET]>=median,'High','Low')
    return d,median

@st.cache_resource(show_spinner='Training models...')
def train(d):
    X,y=d[FEATURES],d['Consumption_Level']; Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    ms={'Logistic Regression':make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,random_state=42)), 'Decision Tree':DecisionTreeClassifier(max_depth=8,random_state=42), 'Random Forest':RandomForestClassifier(n_estimators=100,max_depth=12,random_state=42,n_jobs=-1), 'Gaussian Naive Bayes':GaussianNB()}
    rows=[]; cms={}
    for name,m in ms.items():
        m.fit(Xtr,ytr); p=m.predict(Xte)
        rows.append({'Model':name,'Accuracy':accuracy_score(yte,p),'Precision':precision_score(yte,p,pos_label='High',zero_division=0),'Recall':recall_score(yte,p,pos_label='High',zero_division=0),'F1-score':f1_score(yte,p,pos_label='High',zero_division=0)})
        cms[name]=confusion_matrix(yte,p,labels=['Low','High'])
    results=pd.DataFrame(rows).sort_values('F1-score',ascending=False).reset_index(drop=True); best=results.iloc[0]['Model']
    a,b,c,e=train_test_split(X,d[TARGET],test_size=.2,random_state=42); reg=LinearRegression().fit(a,c); pred=reg.predict(b)
    metrics={'MAE (kW)':mean_absolute_error(e,pred),'MSE':mean_squared_error(e,pred),'RMSE (kW)':mean_squared_error(e,pred)**.5,'R²':r2_score(e,pred)}
    return ms,results,cms,best,reg,metrics

def style(fig,title=None):
    # Explicit dark text on every Plotly label: prevents pale/white labels on light charts.
    fig.update_layout(
        title=dict(text=title, font=dict(color='#172944', size=16)) if title else None,
        paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF',
        font=dict(family='Arial, sans-serif', color='#172944', size=12),
        margin=dict(l=55, r=35, t=58 if title else 35, b=65),
        legend_title_text='',
        legend=dict(font=dict(color='#172944', size=11), title=dict(font=dict(color='#172944'))),
        coloraxis_colorbar=dict(tickfont=dict(color='#172944'), title=dict(font=dict(color='#172944')))
    )
    fig.update_xaxes(
        showgrid=True, gridcolor='#E3E8F0', zerolinecolor='#9AA8BA',
        tickfont=dict(color='#26364D', size=11),
        title_font=dict(color='#26364D', size=12),
        linecolor='#9AA8BA'
    )
    fig.update_yaxes(
        showgrid=True, gridcolor='#E3E8F0', zerolinecolor='#9AA8BA',
        tickfont=dict(color='#26364D', size=11),
        title_font=dict(color='#26364D', size=12),
        linecolor='#9AA8BA'
    )
    # Explicitly darken labels/annotations in heatmaps and other Plotly figures.
    fig.update_annotations(font=dict(color='#172944', size=11))
    return fig

# Load the dataset automatically from the Colab runtime; no in-dashboard upload is needed.
# Preferred filename first, then search common dataset files in /content and the app folder.
preferred = [
    Path('/content/household_power_consumption.txt'),
    Path('/content/household_power_consumption.csv'),
    Path('household_power_consumption.txt'),
    Path('household_power_consumption.csv'),
]
candidates = preferred + sorted(Path('/content').glob('*.txt')) + sorted(Path('/content').glob('*.csv')) + sorted(Path('.').glob('*.txt')) + sorted(Path('.').glob('*.csv'))
data_path = None
seen = set()
for candidate in candidates:
    try:
        resolved = candidate.resolve()
        if resolved in seen or not candidate.is_file() or candidate.name.lower() in {'sample_submission.csv'}:
            continue
        seen.add(resolved)
        # Verify it looks like the household electricity dataset before selecting it.
        with candidate.open('r', encoding='utf-8', errors='ignore') as handle:
            header = handle.readline().strip().replace('\ufeff', '')
        if 'Global_active_power' in header and ('Sub_metering_1' in header or ';' in header):
            data_path = candidate
            break
    except Exception:
        continue

if data_path is None:
    st.error('Household electricity dataset was not found in the Colab runtime.')
    st.markdown('**One-time setup:** upload `household_power_consumption.txt` or `household_power_consumption.csv` to the Colab notebook Files panel (left sidebar). After that, this dashboard loads it automatically; you will not need an upload control inside the dashboard.')
    st.stop()
try:
    raw, name = data_path.read_bytes(), data_path.name
    df, median = clean_data(raw, name)
except Exception as ex:
    st.error(f'Dataset error in {data_path.name}: {ex}')
    st.stop()
st.sidebar.markdown('---')
st.sidebar.caption(f'Dataset\n{len(df):,} valid records\n7 measured variables')
models,results,cms,best,reg,metrics=train(df)
page=st.sidebar.radio('NAVIGATION',['Overview','Statistical Analysis','Visualizations','Usage Patterns','Classification','Prediction'], key='dashboard_navigation')
st.sidebar.markdown('---'); st.sidebar.caption('B.Tech AI & Data Science')

if page=='Overview':
    st.subheader('Dataset Overview'); st.caption('Explore the cleaned household electricity data, summary metrics, interactive charts, and downloadable cleaned records.'); date_range='Not available'
    if 'DateTime' in df and df.DateTime.notna().any(): date_range=f"{df.DateTime.min():%d %b %Y} – {df.DateTime.max():%d %b %Y}"
    a,b,c,d=st.columns(4); a.metric('Records after cleaning',f'{len(df):,}'); b.metric('Average active power',f'{df[TARGET].mean():.3f} kW'); c.metric('Peak active power',f'{df[TARGET].max():.3f} kW'); d.metric('Median active power',f'{median:.3f} kW')
    st.caption(f'Dataset: {name} | Period: {date_range}')
    left,right=st.columns([1.4,1])
    with left:
        mon=df.dropna(subset=['Month']).groupby('Month',as_index=False)[TARGET].mean()
        fig=px.line(mon,x='Month',y=TARGET,markers=True,title='Monthly Average Active Power',color_discrete_sequence=[BLUE]) if not mon.empty else px.histogram(df.sample(min(20000,len(df)),random_state=42),x=TARGET,nbins=45,title='Active Power Distribution',color_discrete_sequence=[BLUE])
        st.plotly_chart(style(fig),use_container_width=True)
    with right:
        counts=df['Consumption_Level'].value_counts().rename_axis('Level').reset_index(name='Records'); st.plotly_chart(style(px.pie(counts,names='Level',values='Records',hole=.45,title='High / Low Consumption Split',color_discrete_sequence=[BLUE,AMBER])),use_container_width=True)
    st.markdown('#### Variable descriptions'); st.dataframe(pd.DataFrame({'Variable':REQ,'Description':['Total active electrical power (target, kW)','Reactive power (kVAR)','Supply voltage (V)','Current intensity (A)','Sub-meter 1 reading (Wh)','Sub-meter 2 reading (Wh)','Sub-meter 3 reading (Wh)']}),use_container_width=True,hide_index=True)
    st.markdown('#### First 10 cleaned rows'); st.dataframe(df.head(10),use_container_width=True); st.download_button('Download cleaned dataset',df.to_csv(index=False).encode(),file_name='household_power_cleaned.csv')
elif page=='Statistical Analysis':
    st.subheader('Statistical Analysis'); st.caption('Select a measured variable, change the confidence level, or adjust the t-test inputs to explore the results.'); variable=st.selectbox('Select variable',REQ); x=df[variable].dropna().astype(float)
    a,b,c,d=st.columns(4); a.metric('Mean',f'{x.mean():.4f}'); b.metric('Median',f'{x.median():.4f}'); c.metric('Std. deviation',f'{x.std():.4f}'); d.metric('Variance',f'{x.var():.4f}')
    st.dataframe(x.describe().to_frame('Value'),use_container_width=True); st.plotly_chart(style(px.histogram(df.sample(min(20000,len(df)),random_state=42),x=variable,nbins=45,title=f'Distribution of {variable}',color_discrete_sequence=[BLUE])),use_container_width=True)
    t1,t2=st.tabs(['Confidence Interval','One-sample t-test'])
    with t1:
        conf=st.selectbox('Confidence level',[.90,.95,.99],index=1); ci=stats.t.interval(conf,df=len(x)-1,loc=x.mean(),scale=stats.sem(x)); st.success(f'{int(conf*100)}% confidence interval: ({ci[0]:.6f}, {ci[1]:.6f})')
    with t2:
        hypo=st.number_input('Hypothesized mean',value=float(x.mean()),format='%.5f'); alpha=st.selectbox('Significance level',[.01,.05,.10],index=1); n=st.slider('Sample size',100,min(10000,max(100,len(x))),min(1000,len(x)),step=100); sample=x.sample(min(n,len(x)),random_state=42); ts,pv=stats.ttest_1samp(sample,popmean=hypo); st.write(f't-statistic: **{ts:.5f}** | p-value: **{pv:.6g}**'); st.info('Reject H₀.' if pv<alpha else 'Fail to reject H₀.')
elif page=='Visualizations':
    st.subheader('Interactive Visualizations'); st.caption('Choose a chart type and variables using the labelled dropdowns. Hover over charts for values and use the Plotly toolbar where available.'); kind=st.selectbox('Choose visualization',['Histogram','Box plot','Correlation heatmap','Scatter plot','Sub-meter comparison']); feat=st.selectbox('Select variable',REQ); sample=df.sample(min(20000,len(df)),random_state=42)
    if kind=='Histogram': fig=px.histogram(sample,x=feat,nbins=45,color_discrete_sequence=[BLUE])
    elif kind=='Box plot': fig=px.box(sample,y=feat,color_discrete_sequence=[AMBER])
    elif kind=='Correlation heatmap': fig=px.imshow(df[REQ].corr(numeric_only=True),text_auto='.2f',aspect='auto',color_continuous_scale='Blues',zmin=-1,zmax=1)
    elif kind=='Scatter plot':
        xcol=st.selectbox('X-axis',FEATURES,index=2); ycol=st.selectbox('Y-axis',[TARGET]+FEATURES,index=0); fig=px.scatter(sample,x=xcol,y=ycol,opacity=.45,trendline='ols',color_discrete_sequence=[STEEL])
    else:
        sm=df[['Sub_metering_1','Sub_metering_2','Sub_metering_3']].mean().rename_axis('Sub-meter').reset_index(name='Mean reading (Wh)'); fig=px.bar(sm,x='Sub-meter',y='Mean reading (Wh)',color='Sub-meter',color_discrete_sequence=COLORS)
    st.plotly_chart(style(fig,kind),use_container_width=True)
elif page=='Usage Patterns':
    st.subheader('Household Usage Patterns'); st.caption('Hover over the interactive charts to inspect hourly, weekday, and monthly usage patterns.')
    if df['Hour'].isna().all(): st.warning('Time-based charts require Date and Time columns in your dataset.')
    else:
        hourly=df.groupby('Hour',as_index=False)[TARGET].mean(); st.plotly_chart(style(px.line(hourly,x='Hour',y=TARGET,markers=True,title='Average Active Power by Hour',color_discrete_sequence=[BLUE])),use_container_width=True)
        if df['Weekday'].notna().any():
            days=df.groupby('Weekday',as_index=False)[TARGET].mean(); order=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']; days['Weekday']=pd.Categorical(days['Weekday'],categories=order,ordered=True); days=days.sort_values('Weekday'); st.plotly_chart(style(px.bar(days,x='Weekday',y=TARGET,title='Average Active Power by Weekday',color='Weekday',color_discrete_sequence=COLORS)),use_container_width=True)
        if df['Month'].notna().any(): st.plotly_chart(style(px.line(df.groupby('Month',as_index=False)[TARGET].mean(),x='Month',y=TARGET,markers=True,title='Monthly Average Active Power',color_discrete_sequence=[AMBER])),use_container_width=True)
elif page=='Classification':
    st.subheader('High / Low Consumption Classification')
    st.caption('Use the model selector below the comparison chart to switch the confusion matrix. All charts and table values are interactive/scrollable where applicable.')
    st.caption('Four classification models are compared on the same test set. The best one is saved and used for predictions.')
    st.success(f'Best model: {best} (highest F1-score on {len(df):,} cleaned records)')
    show=results.copy()
    for col in ['Accuracy','Precision','Recall','F1-score']: show[col]=show[col].round(4)
    st.markdown('#### Model comparison')
    st.dataframe(show,use_container_width=True,hide_index=True)
    left,right=st.columns([1,1])
    with left:
        plot_df=results.melt(id_vars='Model',value_vars=['Accuracy','Precision','Recall','F1-score'],var_name='Metric',value_name='Score')
        fig=px.bar(plot_df,x='Model',y='Score',color='Metric',barmode='group',title='Performance by model',color_discrete_sequence=['#203A60','#C28A17','#5279AC','#7B8BA5'])
        fig.update_layout(xaxis_tickangle=-30,yaxis_range=[0,1.02],legend=dict(orientation='h',y=1.15,x=0),font_size=10)
        st.plotly_chart(style(fig),use_container_width=True)
    with right:
        chosen=st.selectbox('Confusion matrix for',list(models.keys()),index=list(models.keys()).index(best))
        cmfig=px.imshow(cms[chosen],x=['Low','High'],y=['Low','High'],text_auto=',',color_continuous_scale=[[0,'#F2F4F8'],[1,'#203A60']],aspect='auto',labels={'x':'Predicted','y':'Actual','color':'Records'},title=f'Confusion matrix: {chosen}')
        cmfig.update_layout(font=dict(color='#172944', size=11), font_size=11, coloraxis_colorbar=dict(tickfont=dict(color='#172944'), title=dict(font=dict(color='#172944'))))
        cmfig.update_xaxes(tickfont=dict(color='#26364D'), title_font=dict(color='#26364D'))
        cmfig.update_yaxes(tickfont=dict(color='#26364D'), title_font=dict(color='#26364D'))
        st.plotly_chart(style(cmfig),use_container_width=True)
    st.markdown('<div class="note">Global intensity is closely related to active power, so classification scores may be very high because of this physical relationship. This should be considered when interpreting model performance.</div>',unsafe_allow_html=True)
else:
    st.subheader('Consumption Prediction')
    st.info('Change any labelled measurement, then click **Predict active power** to refresh the estimate and consumption class.')
    st.caption('Enter electrical measurements to estimate active power (linear regression) and classify consumption level (best classifier).')
    a,b,c=st.columns(3)
    a.metric('R² score',f"{metrics['R²']:.3f}")
    b.metric('Mean absolute error',f"{metrics['MAE (kW)']:.3f} kW")
    c.metric('Root mean squared error',f"{metrics['RMSE (kW)']:.3f} kW")
    med=df[FEATURES].median()
    left,right=st.columns([1.05,0.95],gap='large')
    with left:
        st.markdown('#### Input measurements')
        st.caption('Defaults are set to the dataset medians.')
        with st.form('prediction_form'):
            x1,x2=st.columns(2)
            with x1:
                reactive=st.number_input('Global reactive power (kVAR)',min_value=0.0,value=float(med['Global_reactive_power']),step=.01,format='%.3f')
                intensity=st.number_input('Current intensity (A)',min_value=0.0,value=float(med['Global_intensity']),step=.1,format='%.2f')
                sub2=st.number_input('Laundry (sub-meter 2) (Wh)',min_value=0.0,value=float(med['Sub_metering_2']),step=1.)
            with x2:
                voltage=st.number_input('Voltage (V)',min_value=0.0,value=float(med['Voltage']),step=1.,format='%.2f')
                sub1=st.number_input('Kitchen (sub-meter 1) (Wh)',min_value=0.0,value=float(med['Sub_metering_1']),step=1.)
                sub3=st.number_input('Water heater and AC (sub-meter 3) (Wh)',min_value=0.0,value=float(med['Sub_metering_3']),step=1.)
            submitted=st.form_submit_button('Predict active power',use_container_width=True)
    with right:
        st.markdown('#### Result')
        if submitted:
            row=pd.DataFrame([[reactive,voltage,intensity,sub1,sub2,sub3]],columns=FEATURES)
            power=max(0.,float(reg.predict(row)[0])); label=str(models[best].predict(row)[0])
        else:
            row=pd.DataFrame([[float(med[x]) for x in FEATURES]],columns=FEATURES)
            power=max(0.,float(reg.predict(row)[0])); label=str(models[best].predict(row)[0])
        st.caption('Estimated active power')
        st.markdown(f'<div style="background:#fff;border:1px solid #DFE4EC;border-left:3px solid #C28A17;border-radius:5px;padding:14px 16px"><div style="font-size:26px;font-weight:700;color:#203A60;font-family:Georgia,serif">{power:.3f} kW</div><div style="font-size:11px;color:#56657A;margin-top:6px">Predicted level: <b>{label}</b> (best model: {best})</div></div>',unsafe_allow_html=True)
        upper=max(float(df[TARGET].quantile(.99)),power,1.)
        fig=go.Figure(go.Indicator(mode='gauge+number',value=power,number={'suffix':' kW','font':{'size':25,'color':'#26364D'}},title={'text':'Estimated active power','font':{'size':12}},gauge={'axis':{'range':[0,upper],'tickfont':{'size':9}},'threshold':{'line':{'color':AMBER,'width':4},'value':median},'bar':{'color':BLUE,'thickness':.25},'steps':[{'range':[0,min(median,upper)],'color':'#E8EDF4'},{'range':[min(median,upper),upper],'color':'#F5E8C8'}],'bgcolor':'white'}))
        fig.update_layout(paper_bgcolor='white',plot_bgcolor='white',height=270,margin=dict(l=12,r=12,t=38,b=8),font_color='#203A60')
        st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})
        st.caption(f'Median threshold in dataset: {median:.3f} kW. This is a model estimate, not a live meter reading.')
st.markdown('<div class="footer">B.Tech AI & Data Science · Statistical Analysis and Prediction of Household Electricity Consumption</div>',unsafe_allow_html=True)
