import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.formula.api as smf
import statsmodels.api as sm
import patsy
from scipy.stats import chi2_contingency, fisher_exact, chi2
from collections import defaultdict
import hashlib
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="OASI Research Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced CSS for academic styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        letter-spacing: -0.02em;
    }
    
    .section-header {
        font-size: 2rem;
        font-weight: 600;
        color: #2c3e50;
        border-bottom: 3px solid;
        border-image: linear-gradient(90deg, #3498db, #9b59b6) 1;
        padding-bottom: 0.5rem;
        margin: 2rem 0 1.5rem 0;
        letter-spacing: -0.01em;
    }
    
    .academic-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        border: 1px solid #e9ecef;
        margin: 1.5rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        margin: 0.5rem 0;
    }
    
    .finding-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(240, 147, 251, 0.3);
    }
    
    .method-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(79, 172, 254, 0.3);
    }
    
    .interpretation-box {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        color: #2c3e50;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(67, 233, 123, 0.3);
        font-weight: 500;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .stSelectbox > div > div {
        background-color: #f8f9fa;
        border: 2px solid #e9ecef;
        border-radius: 8px;
    }
    
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    .research-question {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #e67e22;
        margin: 1rem 0;
        font-style: italic;
        font-weight: 500;
    }
    
    .significance-badge {
        background: #27ae60;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .non-significance-badge {
        background: #95a5a6;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Authentication system
def check_password():
    """Returns `True` if the user had the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        # Define authorized users (in production, store these securely)
        authorized_users = {
            "researcher": "oasiresearch2025",
        }
        
        username = st.session_state["username"]
        password = st.session_state["password"]
        
        if username in authorized_users and authorized_users[username] == password:
            st.session_state["password_correct"] = True
            st.session_state["current_user"] = username
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    # Return True if password is validated
    if st.session_state.get("password_correct", False):
        return True

    # Show login form
    st.markdown("""
    <div style="max-width: 400px; margin: 5rem auto; padding: 2rem; 
                background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
                border-radius: 20px; box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                border: 1px solid #e9ecef;">
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="color: #2c3e50; font-size: 2rem; margin-bottom: 0.5rem;">🏥 OASI Research</h1>
            <h3 style="color: #7f8c8d; font-weight: 400;">Secure Access Portal</h3>
            <div style="width: 60px; height: 3px; background: linear-gradient(90deg, #3498db, #9b59b6); margin: 1rem auto;"></div>
        </div>
    """, unsafe_allow_html=True)
    
    # Login form
    with st.form("login_form"):
        st.markdown("#### 🔐 Please enter your credentials")
        
        username = st.text_input("Username", key="username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", key="password", placeholder="Enter your password")
        
        login_button = st.form_submit_button("🚀 Access Dashboard", on_click=password_entered)
        
        if login_button:
            if st.session_state.get("password_correct", False):
                st.success("✅ Authentication successful! Welcome to the OASI Research Dashboard.")
                st.rerun()
            else:
                st.error("❌ Invalid username or password. Please try again.")
    
    # Information box
    st.markdown("""
    <div style="margin-top: 2rem; padding: 1rem; background: #e8f4fd; 
                border-radius: 10px; border-left: 4px solid #3498db;">
        <h4 style="color: #2c3e50; margin-bottom: 0.5rem;">📋 Access Information</h4>
        <p style="color: #34495e; margin-bottom: 0;">This dashboard contains sensitive medical research data. 
        Access is restricted to authorized personnel only.</p>
        <br>
        <p style="color: #7f8c8d; font-size: 0.9rem; margin-bottom: 0;">
        <strong>Available Accounts:</strong><br>
        • researcher / oasi2024research<br>
        • clinician / medical2024<br>
        • admin / admin2024oasi<br>
        • indranil / indranil2024
        </p>
    </div>
    </div>
    """, unsafe_allow_html=True)
    
    return False

@st.cache_data
def load_and_process_data():
    """Load and process the OASI data."""
    
    # Load data
    path = 'OASI Research Demographics 0708 final - stats copy new.xlsx'
    df3 = pd.read_excel(path, sheet_name='3C TEAR final')
    df3.drop(['Complications', 'Epaq PF issued', 'Epaq recieved'], axis=1, inplace=True)
    
    # Process df4 with MultiIndex columns
    df4_raw = pd.read_excel(path, sheet_name='4th degree tear final', header=[0, 1])
    df4_raw.drop(['EPAQ PF issued', 'EPAQ PF received'], axis=1, inplace=True)
    
    def clean_lbl(x):
        if x is None: return ''
        s = str(x).strip()
        if s.lower().startswith('unnamed'): return ''
        return s
    
    flat_cols = []
    for top, sub in df4_raw.columns:
        top_c, sub_c = clean_lbl(top), clean_lbl(sub)
        if top_c and sub_c: name = f"{top_c}_{sub_c}"
        elif top_c: name = top_c
        elif sub_c: name = sub_c
        else: name = "Unnamed"
        flat_cols.append(name)
    
    seen = defaultdict(int)
    uniq_cols = []
    for name in flat_cols:
        if name in seen:
            seen[name] += 1
            uniq_cols.append(f"{name}.{seen[name]}")
        else:
            seen[name] = 0
            uniq_cols.append(name)
    
    df4 = df4_raw.copy()
    df4.columns = uniq_cols
    df4 = df4.dropna(axis=1, how='all')
    
    # Create categorical variables
    # Age groups
    bins = [-np.inf, 20, 30, 40, np.inf]
    labels = ['<20', '20-30', '30-40', '>40']
    df3['Age_Group'] = pd.cut(pd.to_numeric(df3['Age_at_deliveryDate'], errors='coerce'), bins=bins, labels=labels, right=False)
    df4['Age_Group'] = pd.cut(pd.to_numeric(df4['Age_at_deliveryDate'], errors='coerce'), bins=bins, labels=labels, right=False)
    
    # BMI groups
    bins = [-np.inf, 25, 30, 40, np.inf]
    labels = ['<25', '25-30', '30-40', '>40']
    df3['BMI_Group'] = pd.cut(pd.to_numeric(df3['bmi'], errors='coerce'), bins=bins, labels=labels, right=False)
    df4['BMI_Group'] = pd.cut(pd.to_numeric(df4['bmi'], errors='coerce'), bins=bins, labels=labels, right=False)
    
    # Parity groups
    df3['Parity_num'] = pd.to_numeric(df3['Parity'], errors='coerce')
    df4['Parity_num'] = pd.to_numeric(df4['Parity'], errors='coerce')
    df3['Parity_Group_v2'] = np.select([df3['Parity_num'] < 1, df3['Parity_num'] == 1, df3['Parity_num'] == 2, df3['Parity_num'] > 2], ['<1', '1', '2', '>2'], default=None)
    df4['Parity_Group_v2'] = np.select([df4['Parity_num'] < 1, df4['Parity_num'] == 1, df4['Parity_num'] == 2, df4['Parity_num'] > 2], ['<1', '1', '2', '>2'], default=None)
    
    # Ethnicity groups
    def recode_ethnicity(x):
        x = str(x).strip().upper()
        if x in ["WHITE BRITISH", "ANY OTHER WHITE BACKGROUND", "WHITE IRISH"]:
            return "White"
        elif x in ["PAKISTANI", "BANGLADESHI", "INDIAN", "ANY OTHER ASIAN BACKGROUND"]:
            return "South Asian"
        elif x in ["BLACK AFRICAN"]:
            return "Black"
        elif x in ["ANY OTHER ETHNIC GROUP", "MIXED WHITE & BLACK CARIBBEAN", "ANY OTHER MIXED BACKGROUND", "CHINESE"]:
            return "Other/Mixed"
        elif x in ["NOT STATED", "0"]:
            return "Unknown"
        else:
            return "Unknown"
    
    df4["EthnicOrigin_cat"] = df4["EthnicOrigin"].apply(recode_ethnicity)
    df3["EthnicOrigin_cat"] = df3["EthnicOrigin"].apply(recode_ethnicity)
    
    # Mode of delivery
    df3['MOD_UP'] = df3['MOD'].astype(str).str.strip().str.upper()
    df4['MOD_UP'] = df4['MOD'].astype(str).str.strip().str.upper()
    df3['MOD_ALL'] = df3['MOD_UP']
    df4['MOD_ALL'] = df4['MOD_UP']
    
    # SVD / NVD / VD synonyms
    svd_mask_3 = df3['MOD_UP'].isin({'SVD','NVD','VD','NORMAL VAGINAL DELIVERY','SPONTANEOUS VAGINAL DELIVERY','VAGINAL'})
    svd_mask_4 = df4['MOD_UP'].isin({'SVD','NVD','VD','NORMAL VAGINAL DELIVERY','SPONTANEOUS VAGINAL DELIVERY','VAGINAL'})
    df3.loc[svd_mask_3, 'MOD_ALL'] = 'SVD'
    df4.loc[svd_mask_4, 'MOD_ALL'] = 'SVD'
    
    # Ventouse / vacuum / Kiwi
    vent_mask_3 = df3['MOD_UP'].str.contains(r'VENTOUSE|VACUUM|KIWI', na=False)
    vent_mask_4 = df4['MOD_UP'].str.contains(r'VENTOUSE|VACUUM|KIWI', na=False)
    df3.loc[vent_mask_3, 'MOD_ALL'] = 'VENTOUSE'
    df4.loc[vent_mask_4, 'MOD_ALL'] = 'VENTOUSE'
    
    # Kielland forceps
    kiel_mask_3 = df3['MOD_UP'].str.contains(r'KIELLAND|^KF$|^KFD$', na=False)
    kiel_mask_4 = df4['MOD_UP'].str.contains(r'KIELLAND|^KF$|^KFD$', na=False)
    df3.loc[kiel_mask_3, 'MOD_ALL'] = 'KIELLAND FORCEPS'
    df4.loc[kiel_mask_4, 'MOD_ALL'] = 'KIELLAND FORCEPS'
    
    # NBFD
    nbfd_mask_3 = (df3['MOD_UP'].eq('NBFD')) | (df3['MOD_UP'].str.contains('FORCEPS', na=False) & (~kiel_mask_3))
    nbfd_mask_4 = (df4['MOD_UP'].eq('NBFD')) | (df4['MOD_UP'].str.contains('FORCEPS', na=False) & (~kiel_mask_4))
    df3.loc[nbfd_mask_3, 'MOD_ALL'] = 'NBFD'
    df4.loc[nbfd_mask_4, 'MOD_ALL'] = 'NBFD'
    
    # CS / Caesarean
    cs_mask_3 = df3['MOD_UP'].str.contains(r'\bCS\b|C/S|CAESAREAN|CESAREAN|EMCS|ELCS', na=False)
    cs_mask_4 = df4['MOD_UP'].str.contains(r'\bCS\b|C/S|CAESAREAN|CESAREAN|EMCS|ELCS', na=False)
    df3.loc[cs_mask_3, 'MOD_ALL'] = 'CS'
    df4.loc[cs_mask_4, 'MOD_ALL'] = 'CS'
    
    # Label blanks as 'MISSING'
    df3.loc[df3['MOD_ALL'].eq('') | df3['MOD_ALL'].isin(['NAN','NONE']), 'MOD_ALL'] = 'MISSING'
    df4.loc[df4['MOD_ALL'].eq('') | df4['MOD_ALL'].isin(['NAN','NONE']), 'MOD_ALL'] = 'MISSING'
    
    # Repair type
    df3['Repair_Type'] = df3['Repair '].astype(str).str.strip()
    raw4 = df4['Method of repair EAS'].astype(str).str.strip()
    norm4 = raw4.str.upper().str.replace(r'[^A-Z]', '', regex=True)
    is_overlap_4 = norm4.str.contains('OVERLAP', na=False)
    is_e2e_4 = norm4.str.contains('ENDTOEND', na=False)
    df4['Repair_Type'] = np.select([is_overlap_4, is_e2e_4, ~(is_overlap_4 | is_e2e_4)], ['Overlapping', 'End to end', 'Other/Unclear'], default='Missing')
    
    # Additional processing for analysis
    # Previous OASI flags
    df3['prev_oasi_raw'] = df3['Prev OASI'].astype(str).str.strip().str.upper()
    df4['prev_oasi_raw'] = df4['Previous OASI'].astype(str).str.strip().str.upper()
    yes_vals = {'Y', 'YES', 'TRUE', '1'}
    no_vals = {'N', 'NO', 'NOA', 'N ', '1XSVD'}
    df3['Prev_OASI_Flag'] = np.where(df3['prev_oasi_raw'].isin(yes_vals), 'Yes', np.where(df3['prev_oasi_raw'].isin(no_vals), 'No', None))
    df4['Prev_OASI_Flag'] = np.where(df4['prev_oasi_raw'].isin(yes_vals), 'Yes', np.where(df4['prev_oasi_raw'].isin(no_vals), 'No', None))
    
    # Laxatives and antibiotics
    lx3_raw = df3['Laxatives'].astype(str).str.strip().str.upper()
    lx4_raw = df4['Laxatives'].astype(str).str.strip().str.upper()
    df3['Q12_Lax'] = np.where(lx3_raw.isin(yes_vals), 'Yes', np.where(lx3_raw.isin(no_vals), 'No', 'Missing/Unknown'))
    df4['Q12_Lax'] = np.where(lx4_raw.isin(yes_vals), 'Yes', np.where(lx4_raw.isin(no_vals), 'No', 'Missing/Unknown'))
    
    abx3_raw = df3['Antibiotics '].astype(str).str.strip().str.upper()
    abx4_raw = df4['Antibiotics'].astype(str).str.strip().str.upper()
    df3['Q13_Abx'] = np.where(abx3_raw.isin(yes_vals), 'Yes', np.where(abx3_raw.isin(no_vals), 'No', 'Missing/Unknown'))
    df4['Q13_Abx'] = np.where(abx4_raw.isin(yes_vals), 'Yes', np.where(abx4_raw.isin(no_vals), 'No', 'Missing/Unknown'))
    
    # Complications for df4
    df4['Complication_bin'] = df4['Complication'].astype(str).str.strip().str.upper().map({'Y': 1, 'N': 0})
    
    # Baby weight groups
    df3['Baby_weight_num'] = pd.to_numeric(df3['Baby_weight'], errors='coerce')
    df4['Baby_weight_num'] = pd.to_numeric(df4['Baby_weight'], errors='coerce')
    
    # Convert to kg if values look like grams (>20)
    df3['Baby_weight_kg'] = np.where(df3['Baby_weight_num'] > 20, df3['Baby_weight_num'] / 1000.0, df3['Baby_weight_num'])
    df4['Baby_weight_kg'] = np.where(df4['Baby_weight_num'] > 20, df4['Baby_weight_num'] / 1000.0, df4['Baby_weight_num'])
    
    # Create birthweight groups
    df3['BW_Group'] = np.select([df3['Baby_weight_kg'] < 2.5, (df3['Baby_weight_kg'] >= 2.5) & (df3['Baby_weight_kg'] <= 4.0), df3['Baby_weight_kg'] > 4.0], ['<2.5', '2.5-4', '>4'], default=None)
    df4['BW_Group'] = np.select([df4['Baby_weight_kg'] < 2.5, (df4['Baby_weight_kg'] >= 2.5) & (df4['Baby_weight_kg'] <= 4.0), df4['Baby_weight_kg'] > 4.0], ['<2.5', '2.5-4', '>4'], default=None)
    
    # Complication types
    t = df4['Complication types']
    s = t.astype(str).str.strip()
    is_missing = t.isna() | s.eq('')
    s = s.str.upper()
    s = s.str.replace(r'\bFAEC', 'FEC', regex=True)
    s = s.str.replace(r'FLATAL|FLATAS', 'FLATUS', regex=True)
    s = s.str.replace('INCONTINECE', 'INCONTINENCE', regex=False)
    s = s.where(~is_missing, 'Missing/Unknown').replace({'UNKNOWN': 'Missing/Unknown'})
    df4['Q17_Type'] = s
    
    # Create complication flags
    def yn_bool(series):
        s = series.astype(str).str.strip().str.upper()
        true_set = {"Y","YES","1","TRUE"}
        false_set = {"N","NO","0","FALSE"}
        out = pd.Series(np.nan, index=s.index, dtype="float")
        out[s.isin(true_set)] = 1.0
        out[s.isin(false_set)] = 0.0
        return out
    
    def any_issue_from_flags(df_flags):
        any_true = df_flags.eq(1).any(axis=1)
        all_false_with_obs = (df_flags.eq(0).all(axis=1)) & df_flags.notna().any(axis=1)
        out = pd.Series(np.nan, index=df_flags.index, dtype="float")
        out[any_true] = 1.0
        out[all_false_with_obs] = 0.0
        return out
    
    # Bowel issues
    bowel_im_cols = ['Bowel (immediate issues)_Faecal incontinence', 'Bowel (immediate issues)_Flatus incontinence', 'Bowel (immediate issues)_Urgency of stool']
    bowel_lt_cols = ['Bowel (long term issues)_Faecal incontinence', 'Bowel (long term issues)_Flatus incontinence', 'Bowel (long term issues)_Urgency']
    
    im_bowel_flags = df4[bowel_im_cols].apply(yn_bool)
    im_defer_issue = yn_bool(df4['Bowel (immediate issues)_Able to defer bowels?']).replace({1:0, 0:1})
    im_bowel_flags['inability_to_defer'] = im_defer_issue
    df4["Bowel_IM_any_issue"] = any_issue_from_flags(im_bowel_flags)
    
    lt_bowel_flags = df4[bowel_lt_cols].apply(yn_bool)
    lt_defer_issue = yn_bool(df4['Bowel (long term issues)_Able to defer bowels?']).replace({1:0, 0:1})
    lt_bowel_flags['inability_to_defer'] = lt_defer_issue
    df4["Bowel_LT_any_issue"] = any_issue_from_flags(lt_bowel_flags)
    
    # Urinary issues
    urinary_im_cols = ['Urinary problems (immediate issues)_Urgency', 'Urinary problems (immediate issues)_Frequency', 'Urinary problems (immediate issues)_Leakage', 'Urinary problems (immediate issues)_Leakage on strenuous activity', 'Urinary problems (immediate issues)_Voiding dysfunction']
    urinary_lt_cols = ['Urinary problems (long term issues)_Urgency', 'Urinary problems (long term issues)_Frequency', 'Urinary problems (long term issues)_Leakage', 'Urinary problems (long term issues)_Leakage on strenuous activity', 'Urinary problems (long term issues)_Voiding dysfunction']
    
    im_urine_flags = df4[urinary_im_cols].apply(yn_bool)
    df4["Urine_IM_any_issue"] = any_issue_from_flags(im_urine_flags)
    
    lt_urine_flags = df4[urinary_lt_cols].apply(yn_bool)
    df4["Urine_LT_any_issue"] = any_issue_from_flags(lt_urine_flags)
    
    # Vaginal issues
    vaginal_im_cols = ['Vaginal problems (immediate)_Body image', 'Vaginal problems (immediate)_Dyspareunia', 'Vaginal problems (immediate)_Vaginal lump']
    vaginal_lt_cols = ['Vaginal problems (long term)_Body image', 'Vaginal problems (long term)_Dyspareunia', 'Vaginal problems (long term)_Vaginal lump']
    
    im_vaginal_flags = df4[vaginal_im_cols].apply(yn_bool)
    df4["Vaginal_IM_any_issue"] = any_issue_from_flags(im_vaginal_flags)
    
    lt_vaginal_flags = df4[vaginal_lt_cols].apply(yn_bool)
    df4["Vaginal_LT_any_issue"] = any_issue_from_flags(lt_vaginal_flags)
    
    return df3, df4

def main():
    # Check authentication first
    if not check_password():
        return
    
    # Welcome message for authenticated users
    current_user = st.session_state.get("current_user", "User")
    
    # Header with user info
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown("""
        <div style="margin-bottom: 2rem;">
            <h1 class="main-header">🏥 OASI Research Dashboard</h1>
            <p style="font-size: 1.4rem; color: #34495e; font-weight: 500; margin-bottom: 0.5rem; text-align: center;">
                Obstetric Anal Sphincter Injuries: Comprehensive Statistical Analysis
            </p>
            <p style="font-size: 1rem; color: #7f8c8d; font-style: italic; text-align: center;">
                Interactive Dashboard for 3C and 4th Degree Tear Outcomes Research
            </p>
            <div style="width: 100px; height: 3px; background: linear-gradient(90deg, #3498db, #9b59b6); margin: 1rem auto;"></div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="text-align: right; padding: 1rem; margin-top: 1rem;">
            <p style="color: #7f8c8d; font-size: 0.9rem; margin-bottom: 0.5rem;">Logged in as:</p>
            <p style="color: #2c3e50; font-weight: 600; margin-bottom: 0.5rem;">👤 {current_user}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚪 Logout", key="logout_button"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()
    
    # Load data
    with st.spinner('Loading data...'):
        df3, df4 = load_and_process_data()
    
    # Enhanced sidebar with user info and navigation
    st.sidebar.markdown(f"""
    <div style="text-align: center; padding: 1rem; margin-bottom: 2rem; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 15px; color: white;">
        <h2 style="color: white; font-weight: 600; margin-bottom: 0.5rem;">📊 Research Navigation</h2>
        <p style="color: #ecf0f1; font-size: 0.9rem;">Welcome, {current_user.title()}</p>
        <p style="color: #bdc3c7; font-size: 0.8rem;">Select analysis section below</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main sections
    section = st.sidebar.selectbox(
        "Choose Main Section:",
        ["🏠 Overview", "📋 Part I: Descriptive Statistics", "🔬 Part II: Analytical Statistics", "⚖️ Part III: Comparative Statistics"]
    )
    
    # Sub-pages based on section
    if section == "📋 Part I: Descriptive Statistics":
        page = st.sidebar.selectbox(
            "Select Question Group:",
            ["Q1-5: Demographics", "Q6-9: Delivery & Repair", "Q10-15: Follow-up & Outcomes", "Q16-18: Complications"]
        )
    elif section == "🔬 Part II: Analytical Statistics":
        page = st.sidebar.selectbox(
            "Select Analysis:",
            ["Q1: Future Complications Risk Factors", "Q2: Bowel Issues Risk Factors", "Q3-4: Urinary Issues Risk Factors", "Q5: Vaginal Issues Risk Factors"]
        )
    elif section == "⚖️ Part III: Comparative Statistics":
        page = st.sidebar.selectbox(
            "Select Comparison:",
            ["A-C: 3C vs 4th Degree", "D-F: Repair Type Comparisons"]
        )
    else:
        page = "Overview"
    
    # Route to appropriate page
    if section == "🏠 Overview" or page == "Overview":
        show_overview(df3, df4)
    elif section == "📋 Part I: Descriptive Statistics":
        if page == "Q1-5: Demographics":
            show_part1_demographics(df3, df4)
        elif page == "Q6-9: Delivery & Repair":
            show_part1_delivery_repair(df3, df4)
        elif page == "Q10-15: Follow-up & Outcomes":
            show_part1_followup(df3, df4)
        elif page == "Q16-18: Complications":
            show_part1_complications(df3, df4)
    elif section == "🔬 Part II: Analytical Statistics":
        if page == "Q1: Future Complications Risk Factors":
            show_part2_general(df4)
        elif page == "Q2: Bowel Issues Risk Factors":
            show_part2_bowel(df4)
        elif page == "Q3-4: Urinary Issues Risk Factors":
            show_part2_urinary(df4)
        elif page == "Q5: Vaginal Issues Risk Factors":
            show_part2_vaginal(df4)
    elif section == "⚖️ Part III: Comparative Statistics":
        if page == "A-C: 3C vs 4th Degree":
            show_part3_3c_vs_4th(df3, df4)
        elif page == "D-F: Repair Type Comparisons":
            show_part3_repair_comparisons(df4)
    
    # Sidebar footer with logout and info
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem; 
                background: #f8f9fa; border-radius: 10px; margin-top: 2rem;">
        <p style="color: #7f8c8d; font-size: 0.8rem; margin-bottom: 0.5rem;">OASI Research Dashboard v1.0</p>
        <p style="color: #95a5a6; font-size: 0.7rem;">Secure medical research platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.sidebar.button("🚪 Secure Logout", key="sidebar_logout"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()

def show_overview(df3, df4):
    st.markdown('<h2 class="section-header">🏥 OASI Research Study Overview</h2>', unsafe_allow_html=True)
    
    # Key metrics dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("3C Tears", f"{len(df3)}", help="Total number of 3C tear cases")
    
    with col2:
        st.metric("4th Degree Tears", f"{len(df4)}", help="Total number of 4th degree tear cases")
    
    with col3:
        general_comp_rate = (df4['Complication_bin'] == 1).sum() / df4['Complication_bin'].notna().sum() * 100
        st.metric("Complication Rate", f"{general_comp_rate:.1f}%", help="General complication rate in 4th degree tears")
    
    with col4:
        total_patients = len(df3) + len(df4)
        st.metric("Total Patients", f"{total_patients}", help="Total patients in the study")
    
    st.markdown("---")
    
    # Visual summary of key demographics
    st.markdown("### 📊 Study Population Overview")
    
    # Create a summary visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution comparison
        age_3c = df3['Age_at_deliveryDate'].mean()
        age_4th = df4['Age_at_deliveryDate'].mean()
        
        fig_age = go.Figure()
        fig_age.add_trace(go.Bar(
            name='Mean Age',
            x=['3C Tears', '4th Degree Tears'],
            y=[age_3c, age_4th],
            marker_color=['#3498db', '#e74c3c'],
            text=[f'{age_3c:.1f}', f'{age_4th:.1f}'],
            textposition='outside'
        ))
        fig_age.update_layout(
            title='Mean Age Comparison',
            yaxis_title='Age (years)',
            height=300,
            showlegend=False
        )
        st.plotly_chart(fig_age, use_container_width=True)
    
    with col2:
        # Repair type distribution
        repair_3c = df3['Repair_Type'].value_counts(normalize=True) * 100
        repair_4th = df4['Repair_Type'].value_counts(normalize=True) * 100
        
        fig_repair = go.Figure()
        for i, repair_type in enumerate(['Overlapping', 'End to end']):
            if repair_type in repair_3c.index and repair_type in repair_4th.index:
                fig_repair.add_trace(go.Bar(
                    name=repair_type,
                    x=['3C Tears', '4th Degree Tears'],
                    y=[repair_3c.get(repair_type, 0), repair_4th.get(repair_type, 0)],
                    marker_color=['#3498db', '#e74c3c'][i % 2]
                ))
        
        fig_repair.update_layout(
            title='Repair Type Distribution (%)',
            yaxis_title='Percentage (%)',
            height=300,
            barmode='group'
        )
        st.plotly_chart(fig_repair, use_container_width=True)
    
    # Study description with better formatting
    st.markdown("### 📋 Research Objectives")
    
    tab1, tab2, tab3 = st.tabs(["🎯 Main Questions", "📊 Analysis Methods", "🔍 Key Findings"])
    
    with tab1:
        st.markdown("""
        #### Primary Research Questions:
        
        **Part I - Descriptive Analysis:**
        1. **Demographics**: Age, ethnicity, parity, BMI, baby birthweight distributions
        2. **Clinical factors**: Mode of delivery, episiotomy use, repair techniques
        3. **Outcomes**: Complications, follow-up care, subsequent pregnancies
        
        **Part II - Risk Factor Analysis:**
        1. **General complications**: What predicts any complication?
        2. **Specific complications**: Bowel, urinary, and vaginal issues
        3. **Temporal patterns**: Immediate vs long-term outcomes
        
        **Part III - Comparative Analysis:**
        1. **Tear severity**: 3C vs 4th degree outcomes
        2. **Repair techniques**: End-to-end vs overlapping methods
        """)
    
    with tab2:
        st.markdown("""
        #### Statistical Methods Used:
        
        **Descriptive Statistics:**
        - Frequency distributions and percentages
        - Cross-tabulations by demographic groups
        - Summary statistics (mean, median, SD)
        
        **Analytical Statistics:**
        - Multivariable logistic regression
        - Odds ratios with 95% confidence intervals
        - Likelihood ratio tests
        
        **Comparative Statistics:**
        - Chi-squared tests
        - Fisher's exact tests (for small samples)
        - Contingency table analysis
        """)
    
    with tab3:
        # Calculate key findings dynamically
        # Most significant predictor from general complications
        general_results = run_regression_for_display(df4, "Complication_bin")
        
        st.markdown("""
        #### 🔍 Key Research Findings:
        
        **Demographics:**
        - Similar age distributions between 3C and 4th degree tears
        - Majority of patients are primiparous
        - Overlapping repair is the most common technique
        
        **Risk Factors:**
        """)
        
        if general_results and not general_results.get('regularized', False):
            significant_factors = general_results['odds_ratios']
            if 'Significance' in significant_factors.columns:
                sig_factors = significant_factors[significant_factors['Significance'] == '✓']
                if not sig_factors.empty:
                    for _, row in sig_factors.iterrows():
                        st.write(f"- **{row['Predictor']}** significantly affects complication risk")
                else:
                    st.write("- No individual factors reached statistical significance")
        
        st.markdown("""
        **Repair Technique Comparisons:**
        - No significant differences in most complication types between repair methods
        - End-to-end repair shows some association with vaginal complications
        """)
    
    # Quick navigation guide
    st.markdown("---")
    st.markdown("### 🧭 Navigation Guide")
    
    nav_col1, nav_col2, nav_col3 = st.columns(3)
    
    with nav_col1:
        st.markdown("""
        **📋 Part I: Descriptive Statistics**
        - Q1-5: Demographics
        - Q6-9: Delivery & Repair
        - Q10-15: Follow-up
        - Q16-18: Complications
        """)
    
    with nav_col2:
        st.markdown("""
        **🔬 Part II: Analytical Statistics**
        - Q1: Future Complications Risk Factors
        - Q2: Bowel Issues Risk Factors  
        - Q3-4: Urinary Issues Risk Factors
        - Q5: Vaginal Issues Risk Factors
        """)
    
    with nav_col3:
        st.markdown("""
        **⚖️ Part III: Comparative Statistics**
        - A-C: 3C vs 4th Degree
        - D-F: Repair Comparisons
        """)
    
    st.info("💡 **Tip**: Use the sidebar to navigate between different sections and download CSV files for any table you see!")
    
    # Academic footer
    st.markdown("---")
    st.markdown("""
    <div class="academic-card">
        <h4>📚 Methodology & References</h4>
        <p><strong>Statistical Software:</strong> Python 3.12, pandas, statsmodels, scipy</p>
        <p><strong>Visualization:</strong> Plotly, Streamlit</p>
        <p><strong>Statistical Methods:</strong></p>
        <ul>
            <li>Multivariable logistic regression with likelihood ratio testing</li>
            <li>Chi-squared and Fisher's exact tests for categorical comparisons</li>
            <li>Odds ratios with 95% confidence intervals</li>
            <li>Ridge regularization for model stability when appropriate</li>
        </ul>
        <p><em>Dashboard created for academic research purposes. All statistical analyses follow standard epidemiological methods.</em></p>
    </div>
    """, unsafe_allow_html=True)

def show_demographics(df3, df4):
    st.markdown('<h2 class="section-header">Demographics Analysis</h2>', unsafe_allow_html=True)
    
    # Interactive demographic comparison
    demo_option = st.selectbox(
        "Select demographic variable to analyze:",
        ["Age Groups", "BMI Categories", "Parity Groups", "Ethnicity", "Repair Type"]
    )
    
    if demo_option == "Age Groups":
        create_comparison_chart(df3, df4, 'Age_Group', 'Age Distribution', 'Age Group')
    elif demo_option == "BMI Categories":
        create_comparison_chart(df3, df4, 'BMI_Group', 'BMI Distribution', 'BMI Category')
    elif demo_option == "Parity Groups":
        create_comparison_chart(df3, df4, 'Parity_Group_v2', 'Parity Distribution', 'Parity Group')
    elif demo_option == "Ethnicity":
        create_comparison_chart(df3, df4, 'EthnicOrigin_cat', 'Ethnicity Distribution', 'Ethnic Group')
    elif demo_option == "Repair Type":
        create_comparison_chart(df3, df4, 'Repair_Type', 'Repair Type Distribution', 'Repair Type')
    
    # Summary statistics table
    st.markdown("### 📊 Summary Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**3C Tears**")
        st.write(f"- Mean age: {df3['Age_at_deliveryDate'].mean():.1f} ± {df3['Age_at_deliveryDate'].std():.1f} years")
        st.write(f"- Mean BMI: {df3['bmi'].mean():.1f} ± {df3['bmi'].std():.1f} kg/m²")
        st.write(f"- Primigravida: {(df3['Parity_Group_v2'] == '<1').sum()} ({(df3['Parity_Group_v2'] == '<1').mean()*100:.1f}%)")
    
    with col2:
        st.markdown("**4th Degree Tears**")
        st.write(f"- Mean age: {df4['Age_at_deliveryDate'].mean():.1f} ± {df4['Age_at_deliveryDate'].std():.1f} years")
        st.write(f"- Mean BMI: {df4['bmi'].mean():.1f} ± {df4['bmi'].std():.1f} kg/m²")
        st.write(f"- Primigravida: {(df4['Parity_Group_v2'] == '<1').sum()} ({(df4['Parity_Group_v2'] == '<1').mean()*100:.1f}%)")

def create_comparison_chart(df3, df4, column, title, xlabel):
    """Create an enhanced comparison chart between 3C and 4th degree tears."""
    
    # Check if column exists in both dataframes
    if column not in df3.columns and column not in df4.columns:
        st.error(f"Column '{column}' not found in either dataset")
        return
    
    # Calculate percentages
    df3_counts = df3[column].value_counts(normalize=True) * 100 if column in df3.columns else pd.Series(dtype=float)
    df4_counts = df4[column].value_counts(normalize=True) * 100 if column in df4.columns else pd.Series(dtype=float)
    
    # Combine data
    all_categories = sorted(set(df3_counts.index.tolist() + df4_counts.index.tolist()))
    
    comparison_data = pd.DataFrame({
        '3C Tears (%)': [df3_counts.get(cat, 0) for cat in all_categories],
        '4th Degree Tears (%)': [df4_counts.get(cat, 0) for cat in all_categories]
    }, index=all_categories)
    
    # Create enhanced Plotly chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='3C Tears',
        x=comparison_data.index,
        y=comparison_data['3C Tears (%)'],
        marker_color='#3498db',
        text=comparison_data['3C Tears (%)'].round(1),
        texttemplate='%{text}%',
        textposition='outside'
    ))
    
    fig.add_trace(go.Bar(
        name='4th Degree Tears',
        x=comparison_data.index,
        y=comparison_data['4th Degree Tears (%)'],
        marker_color='#e74c3c',
        text=comparison_data['4th Degree Tears (%)'].round(1),
        texttemplate='%{text}%',
        textposition='outside'
    ))
    
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'color': '#2c3e50'}
        },
        xaxis_title=xlabel,
        yaxis_title='Percentage (%)',
        barmode='group',
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**3C Tears Summary:**")
        if column in df3.columns:
            total_3c = len(df3)
            missing_3c = df3[column].isna().sum()
            st.write(f"- Total cases: {total_3c}")
            st.write(f"- Missing data: {missing_3c} ({missing_3c/total_3c*100:.1f}%)")
            if df3_counts.any():
                most_common_3c = df3_counts.idxmax()
                st.write(f"- Most common: {most_common_3c} ({df3_counts.max():.1f}%)")
        else:
            st.write("Data not available")
    
    with col2:
        st.markdown("**4th Degree Tears Summary:**")
        if column in df4.columns:
            total_4th = len(df4)
            missing_4th = df4[column].isna().sum()
            st.write(f"- Total cases: {total_4th}")
            st.write(f"- Missing data: {missing_4th} ({missing_4th/total_4th*100:.1f}%)")
            if df4_counts.any():
                most_common_4th = df4_counts.idxmax()
                st.write(f"- Most common: {most_common_4th} ({df4_counts.max():.1f}%)")
        else:
            st.write("Data not available")
    
    # Enhanced data table with styling
    st.markdown("#### 📊 Detailed Data Table")
    
    # Add counts alongside percentages
    detailed_data = comparison_data.copy()
    if column in df3.columns:
        detailed_data['3C Count'] = [df3[column].value_counts().get(cat, 0) for cat in all_categories]
    if column in df4.columns:
        detailed_data['4th Count'] = [df4[column].value_counts().get(cat, 0) for cat in all_categories]
    
    # Reorder columns for better display
    if '3C Count' in detailed_data.columns and '4th Count' in detailed_data.columns:
        detailed_data = detailed_data[['3C Count', '3C Tears (%)', '4th Count', '4th Degree Tears (%)']]
    
    st.dataframe(detailed_data.round(1), use_container_width=True)
    
    # Download button
    csv_data = detailed_data.round(1).to_csv()
    st.download_button(
        label="📥 Download Comparison Data CSV",
        data=csv_data,
        file_name=f"{title.replace(' ', '_').replace(':', '').lower()}_comparison.csv",
        mime="text/csv",
        key=f"download_comparison_{title.replace(' ', '_').replace(':', '').lower()}"
    )

def show_complications(df3, df4):
    st.markdown('<h2 class="section-header">Complications Analysis</h2>', unsafe_allow_html=True)
    
    # Complication type selector
    comp_type = st.selectbox(
        "Select complication type:",
        ["General Complications", "Bowel Issues", "Urinary Issues", "Vaginal Issues"]
    )
    
    if comp_type == "General Complications":
        show_general_complications(df4)
    elif comp_type == "Bowel Issues":
        show_specific_complications(df4, "Bowel")
    elif comp_type == "Urinary Issues":
        show_specific_complications(df4, "Urinary")
    elif comp_type == "Vaginal Issues":
        show_specific_complications(df4, "Vaginal")

def show_general_complications(df4):
    """Show general complications analysis."""
    
    st.markdown("### 🔍 General Complications Overview")
    
    # Calculate rates
    total_patients = len(df4)
    complications = (df4['Complication_bin'] == 1).sum()
    comp_rate = complications / df4['Complication_bin'].notna().sum() * 100
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Patients", total_patients)
    with col2:
        st.metric("Patients with Complications", complications)
    with col3:
        st.metric("Complication Rate", f"{comp_rate:.1f}%")
    
    # Complications by repair type
    st.markdown("#### Complications by Repair Type")
    
    comp_by_repair = df4.groupby('Repair_Type')['Complication_bin'].agg(['count', 'sum', 'mean']).reset_index()
    comp_by_repair['rate'] = comp_by_repair['mean'] * 100
    comp_by_repair = comp_by_repair[comp_by_repair['count'] > 5]  # Only show groups with >5 patients
    
    fig = px.bar(
        comp_by_repair, 
        x='Repair_Type', 
        y='rate',
        title='Complication Rate by Repair Type',
        labels={'rate': 'Complication Rate (%)', 'Repair_Type': 'Repair Type'},
        color='rate',
        color_continuous_scale='Reds'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def show_specific_complications(df4, comp_category):
    """Show specific complications analysis."""
    
    st.markdown(f"### 🔍 {comp_category} Complications")
    
    # Define columns based on category
    if comp_category == "Bowel":
        im_var, lt_var = "Bowel_IM_any_issue", "Bowel_LT_any_issue"
        specific_cols_im = ['Bowel (immediate issues)_Faecal incontinence', 'Bowel (immediate issues)_Flatus incontinence', 'Bowel (immediate issues)_Urgency of stool']
        specific_cols_lt = ['Bowel (long term issues)_Faecal incontinence', 'Bowel (long term issues)_Flatus incontinence', 'Bowel (long term issues)_Urgency']
    elif comp_category == "Urinary":
        im_var, lt_var = "Urine_IM_any_issue", "Urine_LT_any_issue"
        specific_cols_im = ['Urinary problems (immediate issues)_Urgency', 'Urinary problems (immediate issues)_Frequency', 'Urinary problems (immediate issues)_Leakage']
        specific_cols_lt = ['Urinary problems (long term issues)_Urgency', 'Urinary problems (long term issues)_Frequency', 'Urinary problems (long term issues)_Leakage']
    else:  # Vaginal
        im_var, lt_var = "Vaginal_IM_any_issue", "Vaginal_LT_any_issue"
        specific_cols_im = ['Vaginal problems (immediate)_Body image', 'Vaginal problems (immediate)_Dyspareunia', 'Vaginal problems (immediate)_Vaginal lump']
        specific_cols_lt = ['Vaginal problems (long term)_Body image', 'Vaginal problems (long term)_Dyspareunia', 'Vaginal problems (long term)_Vaginal lump']
    
    # Calculate rates
    col1, col2 = st.columns(2)
    
    with col1:
        if im_var in df4.columns:
            im_rate = (df4[im_var] == 1).sum() / df4[im_var].notna().sum() * 100
            st.metric(f"Immediate {comp_category} Issues", f"{im_rate:.1f}%")
    
    with col2:
        if lt_var in df4.columns:
            lt_rate = (df4[lt_var] == 1).sum() / df4[lt_var].notna().sum() * 100
            st.metric(f"Long-term {comp_category} Issues", f"{lt_rate:.1f}%")
    
    # Specific symptoms breakdown
    st.markdown(f"#### Specific {comp_category} Symptoms")
    
    # Create symptoms comparison chart
    symptoms_data = []
    
    for col in specific_cols_im:
        if col in df4.columns:
            symptom_name = col.split('_')[-1]
            im_rate = (df4[col].str.upper() == 'Y').sum() / df4[col].notna().sum() * 100
            symptoms_data.append({'Symptom': symptom_name, 'Period': 'Immediate', 'Rate': im_rate})
    
    for col in specific_cols_lt:
        if col in df4.columns:
            symptom_name = col.split('_')[-1]
            lt_rate = (df4[col].str.upper() == 'Y').sum() / df4[col].notna().sum() * 100
            symptoms_data.append({'Symptom': symptom_name, 'Period': 'Long-term', 'Rate': lt_rate})
    
    if symptoms_data:
        symptoms_df = pd.DataFrame(symptoms_data)
        
        fig = px.bar(
            symptoms_df,
            x='Symptom',
            y='Rate',
            color='Period',
            title=f'{comp_category} Symptoms: Immediate vs Long-term',
            labels={'Rate': 'Rate (%)', 'Symptom': 'Symptom Type'},
            barmode='group'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def show_regression_results(df4):
    st.markdown('<h2 class="section-header">Regression Analysis Results</h2>', unsafe_allow_html=True)
    
    # Model selector
    model_type = st.selectbox(
        "Select outcome to analyze:",
        ["General Complications", "Bowel (Immediate)", "Bowel (Long-term)", 
         "Urinary (Immediate)", "Urinary (Long-term)", "Vaginal (Long-term)"]
    )
    
    # Map to variable names
    outcome_mapping = {
        "General Complications": "Complication_bin",
        "Bowel (Immediate)": "Bowel_IM_any_issue",
        "Bowel (Long-term)": "Bowel_LT_any_issue",
        "Urinary (Immediate)": "Urine_IM_any_issue",
        "Urinary (Long-term)": "Urine_LT_any_issue",
        "Vaginal (Long-term)": "Vaginal_LT_any_issue"
    }
    
    outcome_var = outcome_mapping[model_type]
    
    # Run regression analysis
    try:
        model_results = run_regression_for_display(df4, outcome_var)
        
        if model_results:
            st.markdown(f"### 📊 Results for {model_type}")
            
            # Model summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Sample Size", f"{model_results['n_obs']}")
            with col2:
                st.metric("Pseudo R²", f"{model_results['pseudo_r2']:.3f}")
            with col3:
                st.metric("Overall p-value", f"{model_results['llr_p']:.4f}")
            
            # Odds ratios table
            st.markdown("#### Odds Ratios")
            or_df = model_results['odds_ratios']
            
            # Color code significant results
            def highlight_significant(val):
                if '*' in str(val):
                    return 'background-color: #ffeb3b'
                return ''
            
            styled_df = or_df.style.applymap(highlight_significant)
            st.dataframe(styled_df, use_container_width=True)
            
            # Forest plot
            st.markdown("#### Forest Plot")
            create_forest_plot(model_results)
            
        else:
            st.error(f"Could not analyze {model_type} - insufficient data or model convergence issues.")
            
    except Exception as e:
        st.error(f"Error running analysis: {e}")

def run_regression_for_display(df4, outcome_var):
    """Run regression analysis and return formatted results."""
    
    predictors = ["Age_at_deliveryDate", "Baby_weight_num", "Absence_episiotomy", 
                 "Parity_Group_v2", "EthnicOrigin_cat", "Repair_Type", "MOD_ALL"]
    
    # Prepare data
    df4["Absence_episiotomy"] = df4["Episotomy"].astype(str).str.strip().str.upper().map({"N": 1, "Y": 0, "YES": 0})
    df4['Baby_weight_num'] = pd.to_numeric(df4['Baby_weight'], errors='coerce')
    
    dfm = df4[[outcome_var] + predictors].copy()
    dfm = dfm.dropna(subset=[outcome_var])
    dfm["MOD_ALL"] = dfm["MOD_ALL"].replace({"KIELLAND FORCEPS": "NBFD"})
    dfm_full = dfm.dropna(subset=predictors)
    
    if dfm_full.empty or dfm_full[outcome_var].nunique() < 2:
        return None
    
    formula = (
        f"{outcome_var} ~ "
        "Age_at_deliveryDate + Baby_weight_num + Absence_episiotomy + "
        "C(Parity_Group_v2, Treatment(reference='1')) + "
        "C(EthnicOrigin_cat, Treatment(reference='White')) + "
        "C(Repair_Type, Treatment(reference='Overlapping')) + "
        "C(MOD_ALL, Treatment(reference='SVD'))"
    )
    
    try:
        y, X = patsy.dmatrices(formula, data=dfm_full, return_type='dataframe')
        zero_cols = [c for c in X.columns if (X[c].abs().sum() == 0)]
        X = X.drop(columns=zero_cols, errors='ignore')
        X = X.loc[:, ~X.columns.duplicated()]
        
        try:
            model = sm.Logit(y, X).fit(disp=0)
            regularized = False
        except:
            model = sm.Logit(y, X).fit_regularized(alpha=1.0, L1_wt=0.0, disp=0)
            regularized = True
        
        if regularized:
            # Handle regularized model
            or_data = []
            for param in model.params.index:
                if param != 'Intercept':
                    or_val = np.exp(model.params[param])
                    clean_name = get_clean_predictor_name(param)
                    or_data.append({
                        'Predictor': clean_name,
                        'OR': or_val,
                        'OR (95% CI)': f"{or_val:.2f} (regularized)",
                        'p_value': np.nan,
                        'CI_lower': np.nan,
                        'CI_upper': np.nan
                    })
            
            or_df = pd.DataFrame(or_data)
            return {
                'n_obs': len(dfm_full),
                'pseudo_r2': np.nan,
                'llr_p': np.nan,
                'odds_ratios': or_df[['Predictor', 'OR (95% CI)']],
                'forest_data': or_df,
                'regularized': True
            }
        
        # Calculate LR test for standard model
        null_model = sm.Logit(y, np.ones((len(y), 1))).fit(disp=0)
        lr_stat = 2 * (model.llf - null_model.llf)
        lr_p = chi2.sf(lr_stat, X.shape[1] - 1)
        
        # Format results
        params = model.params.drop('Intercept', errors='ignore')
        conf = model.conf_int().drop('Intercept', errors='ignore')
        p_values = model.pvalues.drop('Intercept', errors='ignore')
        
        or_data = []
        for param in params.index:
            or_val = np.exp(params[param])
            ci_low = np.exp(conf.loc[param, 0])
            ci_high = np.exp(conf.loc[param, 1])
            p_val = p_values[param]
            
            # Significance markers
            if p_val < 0.001:
                sig = '***'
            elif p_val < 0.01:
                sig = '**'
            elif p_val < 0.05:
                sig = '*'
            else:
                sig = ''
            
            clean_name = get_clean_predictor_name(param)
            or_data.append({
                'Predictor': clean_name,
                'OR': or_val,
                'CI_lower': ci_low,
                'CI_upper': ci_high,
                'p_value': p_val,
                'OR (95% CI)': f"{or_val:.2f} ({ci_low:.2f}-{ci_high:.2f}){sig}",
                'Significance': '✓' if p_val < 0.05 else ''
            })
        
        or_df = pd.DataFrame(or_data)
        
        return {
            'n_obs': len(dfm_full),
            'pseudo_r2': model.prsquared,
            'llr_p': lr_p,
            'odds_ratios': or_df[['Predictor', 'OR (95% CI)', 'Significance']],
            'forest_data': or_df,
            'regularized': False
        }
        
    except Exception as e:
        st.error(f"Error in regression analysis: {e}")
        return None

def get_clean_predictor_name(param):
    """Clean up predictor names for display."""
    predictor_names = {
        'Age_at_deliveryDate': 'Age (per year)',
        'Baby_weight_num': 'Baby weight (per kg)',
        'Absence_episiotomy': 'No episiotomy',
        "C(Parity_Group_v2, Treatment(reference='1'))[T.<1]": 'Parity <1 (vs 1)',
        "C(Parity_Group_v2, Treatment(reference='1'))[T.2]": 'Parity 2 (vs 1)',
        "C(EthnicOrigin_cat, Treatment(reference='White'))[T.South Asian]": 'South Asian (vs White)',
        "C(EthnicOrigin_cat, Treatment(reference='White'))[T.Black]": 'Black (vs White)',
        "C(EthnicOrigin_cat, Treatment(reference='White'))[T.Other/Mixed]": 'Other/Mixed (vs White)',
        "C(EthnicOrigin_cat, Treatment(reference='White'))[T.Unknown]": 'Unknown (vs White)',
        "C(Repair_Type, Treatment(reference='Overlapping'))[T.End to end]": 'End-to-end repair (vs Overlapping)',
        "C(Repair_Type, Treatment(reference='Overlapping'))[T.Other/Unclear]": 'Other repair (vs Overlapping)',
        "C(MOD_ALL, Treatment(reference='SVD'))[T.NBFD]": 'NBFD (vs SVD)',
        "C(MOD_ALL, Treatment(reference='SVD'))[T.VENTOUSE]": 'Ventouse (vs SVD)'
    }
    return predictor_names.get(param, param)

def create_forest_plot(model_results):
    """Create an enhanced interactive forest plot."""
    
    forest_data = model_results['forest_data']
    
    if model_results.get('regularized', False):
        st.warning("Forest plot not available for regularized models (no confidence intervals)")
        return
    
    # Filter out any rows with invalid CI data
    valid_data = forest_data.dropna(subset=['CI_lower', 'CI_upper'])
    
    if valid_data.empty:
        st.warning("No valid confidence interval data for forest plot")
        return
    
    # Sort by OR for better visualization
    valid_data = valid_data.sort_values('OR')
    
    fig = go.Figure()
    
    # Color scheme for significance
    colors = ['#e74c3c' if p < 0.05 else '#3498db' for p in valid_data['p_value']]
    
    # Add error bars
    fig.add_trace(go.Scatter(
        x=valid_data['OR'],
        y=valid_data['Predictor'],
        error_x=dict(
            type='data',
            symmetric=False,
            array=valid_data['CI_upper'] - valid_data['OR'],
            arrayminus=valid_data['OR'] - valid_data['CI_lower'],
            thickness=3,
            width=8
        ),
        mode='markers',
        marker=dict(
            size=12,
            color=colors,
            line=dict(width=2, color='white')
        ),
        text=[f"<b>{row['Predictor']}</b><br>OR: {row['OR']:.2f}<br>95% CI: ({row['CI_lower']:.2f} - {row['CI_upper']:.2f})<br>p-value: {row['p_value']:.4f}" 
              for _, row in valid_data.iterrows()],
        hovertemplate='%{text}<extra></extra>',
        name=""
    ))
    
    # Add vertical line at OR = 1 (no effect)
    fig.add_vline(x=1, line_dash="dash", line_color="gray", line_width=2)
    
    # Add annotations for reference lines
    fig.add_annotation(
        x=1, y=len(valid_data),
        text="No Effect",
        showarrow=False,
        yshift=10,
        font=dict(size=10, color="gray")
    )
    
    fig.update_layout(
        title={
            'text': "Forest Plot: Multivariable Risk Factor Analysis",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#2c3e50', 'family': 'Inter'}
        },
        xaxis_title={
            'text': "Odds Ratio (95% Confidence Interval)",
            'font': {'size': 14, 'color': '#34495e'}
        },
        yaxis_title={
            'text': "Risk Factors",
            'font': {'size': 14, 'color': '#34495e'}
        },
        height=max(500, len(valid_data) * 50 + 150),
        showlegend=False,
        xaxis=dict(
            type="log",
            gridcolor='#ecf0f1',
            gridwidth=1,
            showgrid=True,
            zeroline=False,
            tickfont=dict(size=12, color='#2c3e50')
        ),
        yaxis=dict(
            gridcolor='#ecf0f1',
            gridwidth=1,
            showgrid=True,
            tickfont=dict(size=12, color='#2c3e50')
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=250, r=80, t=100, b=80),
        font=dict(family='Inter', size=12, color='#2c3e50')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Academic interpretation guide
    with st.expander("📖 Statistical Interpretation Guide"):
        st.markdown("""
        <div class="method-box">
            <h4>Forest Plot Reading Guide</h4>
            <ul>
                <li><strong style="color: #e74c3c;">Red markers</strong>: Statistically significant associations (p < 0.05)</li>
                <li><strong style="color: #3498db;">Blue markers</strong>: Non-significant associations (p ≥ 0.05)</li>
                <li><strong>Horizontal error bars</strong>: 95% confidence intervals</li>
                <li><strong>Vertical reference line</strong>: No effect (OR = 1.0)</li>
                <li><strong>Left of reference</strong>: Protective factors (OR < 1.0)</li>
                <li><strong>Right of reference</strong>: Risk factors (OR > 1.0)</li>
            </ul>
            
            <h4>Clinical Interpretation</h4>
            <p>Odds ratios represent the relative odds of developing complications. For example:</p>
            <ul>
                <li>OR = 2.0 means twice the odds of complications</li>
                <li>OR = 0.5 means half the odds of complications</li>
                <li>Confidence intervals crossing 1.0 indicate non-significance</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Download forest plot data
    forest_csv = valid_data[['Predictor', 'OR', 'CI_lower', 'CI_upper', 'p_value']].to_csv(index=False)
    st.download_button(
        label="📥 Download Forest Plot Data CSV",
        data=forest_csv,
        file_name="forest_plot_data.csv",
        mime="text/csv",
        key=f"download_forest_{hash(str(valid_data['Predictor'].tolist()))}"
    )

def show_comparative_stats(df4):
    st.markdown('<h2 class="section-header">Comparative Statistics</h2>', unsafe_allow_html=True)
    
    st.markdown("### 🔬 Statistical Comparisons: End-to-end vs Overlapping Repair")
    
    # Filter to main repair types
    df_comp = df4[df4['Repair_Type'].isin(['Overlapping', 'End to end'])].copy()
    
    # Define comparisons
    comparisons = {
        "Immediate Bowel": "Bowel_IM_any_issue",
        "Long-term Bowel": "Bowel_LT_any_issue", 
        "Immediate Urinary": "Urine_IM_any_issue",
        "Long-term Urinary": "Urine_LT_any_issue",
        "Immediate Vaginal": "Vaginal_IM_any_issue",
        "Long-term Vaginal": "Vaginal_LT_any_issue"
    }
    
    results_data = []
    
    for comp_name, comp_var in comparisons.items():
        if comp_var in df_comp.columns:
            # Create contingency table
            contingency = pd.crosstab(df_comp['Repair_Type'], df_comp[comp_var])
            
            if contingency.shape == (2, 2):
                # Run statistical test
                chi2_stat, chi2_p, _, expected = chi2_contingency(contingency)
                
                if (expected < 5).any():
                    # Use Fisher's exact test
                    odds_ratio, fisher_p = fisher_exact(contingency)
                    test_used = "Fisher's Exact"
                    p_value = fisher_p
                else:
                    # Use Chi-squared test
                    test_used = "Chi-squared"
                    p_value = chi2_p
                
                # Calculate rates for each group
                overlapping_rate = contingency.loc['Overlapping', 1] / contingency.loc['Overlapping'].sum() * 100
                endtoend_rate = contingency.loc['End to end', 1] / contingency.loc['End to end'].sum() * 100
                
                results_data.append({
                    'Complication': comp_name,
                    'Overlapping Rate': f"{overlapping_rate:.1f}%",
                    'End-to-end Rate': f"{endtoend_rate:.1f}%",
                    'Test Used': test_used,
                    'p-value': f"{p_value:.4f}",
                    'Significant': "Yes" if p_value < 0.05 else "No"
                })
    
    # Display results table
    if results_data:
        results_df = pd.DataFrame(results_data)
        
        # Color code significant results
        def highlight_significant(row):
            if row['Significant'] == 'Yes':
                return ['background-color: #ffeb3b'] * len(row)
            return [''] * len(row)
        
        styled_results = results_df.style.apply(highlight_significant, axis=1)
        st.dataframe(styled_results, use_container_width=True)
        
        # Summary
        significant_count = (results_df['Significant'] == 'Yes').sum()
        st.markdown(f"**Summary:** {significant_count} out of {len(results_df)} comparisons showed statistically significant differences (p < 0.05)")
    
    # Interactive comparison chart
    st.markdown("#### Interactive Comparison")
    
    selected_comparison = st.selectbox(
        "Select complication type for detailed view:",
        list(comparisons.keys())
    )
    
    if selected_comparison:
        comp_var = comparisons[selected_comparison]
        if comp_var in df_comp.columns:
            create_comparison_visualization(df_comp, comp_var, selected_comparison)

def create_comparison_visualization(df_comp, comp_var, comp_name):
    """Create an interactive comparison visualization."""
    
    # Calculate rates by repair type
    rates_by_repair = df_comp.groupby('Repair_Type')[comp_var].agg(['count', 'sum', 'mean']).reset_index()
    rates_by_repair['rate'] = rates_by_repair['mean'] * 100
    
    # Create bar chart
    fig = px.bar(
        rates_by_repair,
        x='Repair_Type',
        y='rate',
        title=f'{comp_name} Complication Rate by Repair Type',
        labels={'rate': 'Complication Rate (%)', 'Repair_Type': 'Repair Type'},
        color='rate',
        color_continuous_scale='RdYlBu_r',
        text='rate'
    )
    
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(height=400, showlegend=False)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show contingency table
    st.markdown("#### Contingency Table")
    contingency = pd.crosstab(df_comp['Repair_Type'], df_comp[comp_var], margins=True)
    st.dataframe(contingency, use_container_width=True)
    
    # Download button for contingency table
    contingency_csv = contingency.to_csv()
    st.download_button(
        label="📥 Download Contingency Table CSV",
        data=contingency_csv,
        file_name=f"{selected_comparison.replace(' ', '_').lower()}_contingency_table.csv",
        mime="text/csv",
        key=f"download_contingency_{selected_comparison.replace(' ', '_').lower()}"
    )

# =============================================================================
# PART I: DESCRIPTIVE STATISTICS PAGES
# =============================================================================

def show_part1_demographics(df3, df4):
    st.markdown('<h2 class="section-header">Part I: Questions 1-5 - Demographics</h2>', unsafe_allow_html=True)
    
    st.markdown("### 📊 Demographic Stratification Analysis")
    
    # Question selector
    question = st.selectbox(
        "Select demographic question:",
        ["Q1: Age Groups", "Q2: Ethnicity", "Q3: Parity", "Q4: BMI", "Q5: Baby Birthweight"]
    )
    
    if question == "Q1: Age Groups":
        st.markdown("#### Q1: Stratification by Age (Groups: <20, 20-30, 30-40, >40)")
        create_comparison_chart(df3, df4, 'Age_Group', 'Age Distribution', 'Age Group')
        
    elif question == "Q2: Ethnicity":
        st.markdown("#### Q2: Stratification by Ethnicity")
        create_comparison_chart(df3, df4, 'EthnicOrigin_cat', 'Ethnicity Distribution', 'Ethnic Group')
        
    elif question == "Q3: Parity":
        st.markdown("#### Q3: Stratification by Parity (Groups: <1, 1, 2, >2)")
        create_comparison_chart(df3, df4, 'Parity_Group_v2', 'Parity Distribution', 'Parity Group')
        
    elif question == "Q4: BMI":
        st.markdown("#### Q4: Stratification by BMI (Groups: <25, 25-30, 30-40, >40)")
        create_comparison_chart(df3, df4, 'BMI_Group', 'BMI Distribution', 'BMI Category')
        
    elif question == "Q5: Baby Birthweight":
        st.markdown("#### Q5: Stratification by Baby Birthweight (Groups: <2.5, 2.5-4, >4 kg)")
        create_comparison_chart(df3, df4, 'BW_Group', 'Baby Birthweight Distribution', 'Weight Category')

def show_part1_delivery_repair(df3, df4):
    st.markdown('<h2 class="section-header">Part I: Questions 6-9 - Delivery & Repair</h2>', unsafe_allow_html=True)
    
    question = st.selectbox(
        "Select question:",
        ["Q6: Mode of Delivery", "Q7: Episiotomy by Delivery Mode", "Q8: Previous 3rd Degree Tear", "Q9: Type of Repair"]
    )
    
    if question == "Q6: Mode of Delivery":
        st.markdown("#### Q6: Mode of Delivery Distribution")
        
        # Calculate MOD distribution
        mod_3c = df3['MOD_ALL'].value_counts(normalize=True) * 100
        mod_4th = df4['MOD_ALL'].value_counts(normalize=True) * 100
        
        mod_comparison = pd.DataFrame({
            '3C Tears (%)': mod_3c,
            '4th Degree Tears (%)': mod_4th
        }).fillna(0)
        
        # Create chart
        fig = go.Figure()
        fig.add_trace(go.Bar(name='3C Tears', x=mod_comparison.index, y=mod_comparison['3C Tears (%)'], marker_color='lightblue'))
        fig.add_trace(go.Bar(name='4th Degree Tears', x=mod_comparison.index, y=mod_comparison['4th Degree Tears (%)'], marker_color='darkblue'))
        fig.update_layout(title='Mode of Delivery Distribution', xaxis_title='Delivery Mode', yaxis_title='Percentage (%)', barmode='group')
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(mod_comparison.round(1), use_container_width=True)
        
        # Download button
        csv_data = mod_comparison.round(1).to_csv()
        st.download_button(
            label="📥 Download MOD Distribution CSV",
            data=csv_data,
            file_name="mode_of_delivery_distribution.csv",
            mime="text/csv",
            key="download_mod_distribution"
        )
        
    elif question == "Q8: Previous 3rd Degree Tear":
        st.markdown("#### Q8: Previous 3rd Degree Tear History")
        
        # Calculate previous OASI rates
        prev_3c = (df3['Prev_OASI_Flag'] == 'Yes').sum() / df3['Prev_OASI_Flag'].notna().sum() * 100
        prev_4th = (df4['Prev_OASI_Flag'] == 'Yes').sum() / df4['Prev_OASI_Flag'].notna().sum() * 100
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("3C Tears", f"{prev_3c:.1f}%", help="Percentage with previous 3rd degree tear")
        with col2:
            st.metric("4th Degree Tears", f"{prev_4th:.1f}%", help="Percentage with previous 3rd degree tear")
            
    elif question == "Q9: Type of Repair":
        st.markdown("#### Q9: Type of Repair Distribution")
        create_comparison_chart(df3, df4, 'Repair_Type', 'Repair Type Distribution', 'Repair Type')

def show_part1_followup(df3, df4):
    st.markdown('<h2 class="section-header">Part I: Questions 10-15 - Follow-up & Outcomes</h2>', unsafe_allow_html=True)
    
    question = st.selectbox(
        "Select question:",
        ["Q10: Subsequent Deliveries", "Q11: Type of Repair", "Q12: Laxatives", "Q13: Antibiotics", "Q14: Perineal Clinic", "Q15: Endoanal Scanning"]
    )
    
    if question == "Q12: Laxatives":
        st.markdown("#### Q12: Laxatives in Post-op Advice")
        
        # Calculate laxative rates
        lax_3c = (df3['Q12_Lax'] == 'Yes').sum() / len(df3) * 100
        lax_4th = (df4['Q12_Lax'] == 'Yes').sum() / len(df4) * 100
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("3C Tears", f"{lax_3c:.1f}%", help="Percentage receiving laxatives")
        with col2:
            st.metric("4th Degree Tears", f"{lax_4th:.1f}%", help="Percentage receiving laxatives")
            
    elif question == "Q13: Antibiotics":
        st.markdown("#### Q13: Antibiotics in Post-op Advice")
        
        # Calculate antibiotic rates
        abx_3c = (df3['Q13_Abx'] == 'Yes').sum() / len(df3) * 100
        abx_4th = (df4['Q13_Abx'] == 'Yes').sum() / len(df4) * 100
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("3C Tears", f"{abx_3c:.1f}%", help="Percentage receiving antibiotics")
        with col2:
            st.metric("4th Degree Tears", f"{abx_4th:.1f}%", help="Percentage receiving antibiotics")

def show_part1_complications(df3, df4):
    st.markdown('<h2 class="section-header">Part I: Questions 16-18 - Complications</h2>', unsafe_allow_html=True)
    
    question = st.selectbox(
        "Select question:",
        ["Q16: General Complications", "Q17: Types of Complications", "Q18: Further Therapy"]
    )
    
    if question == "Q16: General Complications":
        st.markdown("#### Q16: General Complications Rate")
        
        # Calculate general complication rates
        comp_4th = (df4['Complication_bin'] == 1).sum() / df4['Complication_bin'].notna().sum() * 100
        
        st.metric("4th Degree Tears", f"{comp_4th:.1f}%", help="General complication rate")
        st.info("Note: 3C tear complication data is not available in the dataset")
        
    elif question == "Q17: Types of Complications":
        st.markdown("#### Q17: Specific Complication Types (4th Degree Only)")
        
        # Get complication types
        comp_types = df4['Q17_Type'].value_counts(normalize=True) * 100
        comp_types = comp_types[comp_types > 1]  # Only show >1%
        
        fig = px.bar(
            x=comp_types.values, 
            y=comp_types.index,
            orientation='h',
            title='Types of Complications (%)',
            labels={'x': 'Percentage (%)', 'y': 'Complication Type'}
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Download button for complication types
        comp_types_df = pd.DataFrame({
            'Complication Type': comp_types.index,
            'Percentage (%)': comp_types.values
        })
        comp_csv = comp_types_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Complication Types CSV",
            data=comp_csv,
            file_name="complication_types_distribution.csv",
            mime="text/csv",
            key="download_complication_types"
        )

# =============================================================================
# PART II: ANALYTICAL STATISTICS PAGES
# =============================================================================

def show_part2_general(df4):
    st.markdown('<h2 class="section-header">Part II: Q1 - Future Complications Risk Factors</h2>', unsafe_allow_html=True)
    
    # Research question with clean formatting
    st.info("""
    🎯 **Research Question 1:**
    
    **Objective:** Identify relationships between maternal age, parity, ethnicity, baby birthweight, 
    absence of episiotomy, type of repair and mode of delivery for having future complications after 3C and 4th degree tears.
    
    **Methodology:** Multivariable logistic regression with univariate and multivariate analysis. 
    Complications treated as binary outcome (Yes/No). Likelihood ratio testing used for model comparison.
    
    **Hypothesis:** Certain maternal and clinical factors may predict increased risk of complications following OASI repair.
    
    **Note:** Analysis limited to 4th degree tears due to data availability in the 3C dataset.
    """)
    
    # Run and display regression
    model_results = run_regression_for_display(df4, "Complication_bin")
    
    if model_results:
        # Model summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Sample Size", f"{model_results['n_obs']}")
        with col2:
            st.metric("Pseudo R²", f"{model_results['pseudo_r2']:.3f}")
        with col3:
            st.metric("Overall p-value", f"{model_results['llr_p']:.4f}")
        
        # Results table
        st.markdown("#### Odds Ratios")
        st.dataframe(model_results['odds_ratios'], use_container_width=True)
        
        # Download button for odds ratios
        or_csv = model_results['odds_ratios'].to_csv(index=False)
        st.download_button(
            label="📥 Download Odds Ratios CSV",
            data=or_csv,
            file_name="general_complications_odds_ratios.csv",
            mime="text/csv",
            key="download_general_complications_or"
        )
        
        # Forest plot
        st.markdown("#### Forest Plot")
        create_forest_plot(model_results)

def show_part2_bowel(df4):
    st.markdown('<h2 class="section-header">Part II: Q2 - Bowel Issues Risk Factors</h2>', unsafe_allow_html=True)
    
    # Research question with clean formatting
    st.info("""
    🎯 **Research Question 2:**
    
    **Objective:** Determine relationships between maternal age, parity, ethnicity, baby birthweight, 
    absence of episiotomy, type of repair and mode of delivery for having immediate and late bowel issues.
    
    **Outcome Definition:** Any bowel complication including faecal incontinence, flatus incontinence, 
    urgency of stool, or inability to defer bowels (Y in any one of these problems).
    
    **Methodology:** Separate multivariable logistic regression models for immediate and long-term outcomes.
    
    **Clinical Relevance:** Bowel dysfunction is a major concern following OASI repair, affecting quality of life.
    """)
    
    # Tabs for immediate vs long-term
    tab1, tab2 = st.tabs(["Immediate Bowel Issues", "Long-term Bowel Issues"])
    
    with tab1:
        model_results = run_regression_for_display(df4, "Bowel_IM_any_issue")
        if model_results:
            display_regression_results(model_results, "Immediate Bowel Issues")
    
    with tab2:
        model_results = run_regression_for_display(df4, "Bowel_LT_any_issue")
        if model_results:
            display_regression_results(model_results, "Long-term Bowel Issues")

def show_part2_urinary(df4):
    st.markdown('<h2 class="section-header">Part II: Q3-4 - Urinary Issues Risk Factors</h2>', unsafe_allow_html=True)
    
    # Research question with clean formatting
    st.info("""
    🎯 **Research Questions 3 & 4:**
    
    **Objective:** Investigate relationships between maternal age, parity, ethnicity, baby birthweight, 
    absence of episiotomy, type of repair and mode of delivery for having immediate and late urinary issues.
    
    **Outcome Definition:** Any urinary complication including urgency, frequency, leakage, 
    leakage on strenuous activity, or voiding dysfunction (Y in any one of these problems).
    
    **Methodology:** Separate multivariable logistic regression models for immediate and long-term outcomes.
    
    **Clinical Relevance:** Urinary dysfunction can significantly impact women's daily activities and quality of life post-OASI.
    
    **Note:** Questions 3 and 4 are identical in the original protocol, so analyzed together.
    """)
    
    # Tabs for immediate vs long-term
    tab1, tab2 = st.tabs(["Immediate Urinary Issues", "Long-term Urinary Issues"])
    
    with tab1:
        model_results = run_regression_for_display(df4, "Urine_IM_any_issue")
        if model_results:
            display_regression_results(model_results, "Immediate Urinary Issues")
    
    with tab2:
        model_results = run_regression_for_display(df4, "Urine_LT_any_issue")
        if model_results:
            display_regression_results(model_results, "Long-term Urinary Issues")

def show_part2_vaginal(df4):
    st.markdown('<h2 class="section-header">Part II: Q5 - Vaginal Issues Risk Factors</h2>', unsafe_allow_html=True)
    
    # Research question with clean formatting
    st.info("""
    🎯 **Research Question 5:**
    
    **Objective:** Examine relationships between maternal age, parity, ethnicity, baby birthweight, 
    absence of episiotomy, type of repair and mode of delivery for having immediate and late vaginal issues.
    
    **Outcome Definition:** Any vaginal complication including body image issues, dyspareunia, 
    or vaginal lump (Y in any one of these problems).
    
    **Methodology:** Separate multivariable logistic regression models for immediate and long-term outcomes.
    
    **Clinical Relevance:** Vaginal complications, particularly dyspareunia, can significantly affect 
    intimate relationships and psychological well-being.
    
    **Special Consideration:** Some models may require regularization due to sparse data in certain categories.
    """)
    
    # Tabs for immediate vs long-term
    tab1, tab2 = st.tabs(["Immediate Vaginal Issues", "Long-term Vaginal Issues"])
    
    with tab1:
        model_results = run_regression_for_display(df4, "Vaginal_IM_any_issue")
        if model_results:
            display_regression_results(model_results, "Immediate Vaginal Issues")
        else:
            st.warning("Regularized model used - limited results available due to data sparsity")
    
    with tab2:
        model_results = run_regression_for_display(df4, "Vaginal_LT_any_issue")
        if model_results:
            display_regression_results(model_results, "Long-term Vaginal Issues")

# =============================================================================
# PART III: COMPARATIVE STATISTICS PAGES
# =============================================================================

def show_part3_3c_vs_4th(df3, df4):
    st.markdown('<h2 class="section-header">Part III: Questions A-C - 3C vs 4th Degree Comparison</h2>', unsafe_allow_html=True)
    
    # Research questions overview
    st.info("""
    🎯 **Research Questions A, B, C:**
    
    **A.** Any significant difference between the rate of immediate and late bowel complications between 3C and 4th degree tears?
    
    **B.** Any significant difference between the rate of immediate and late urinary complications between 3C and 4th degree tears?
    
    **C.** Any significant difference between the rate of immediate and late vaginal complications between 3C and 4th degree tears?
    """)
    
    # Data limitation explanation
    st.error("""
    ⚠️ **Data Limitation Notice**
    
    Questions A, B, and C cannot be answered directly because:
    - The 3C tear dataset does not contain detailed immediate/late complication data
    - Only the 4th degree tear dataset has the specific bowel, urinary, and vaginal issue columns
    
    **Alternative Analysis Available:** General demographic and clinical factor comparisons
    """)
    
    # Available comparisons with enhanced presentation
    st.markdown("### 📊 Available Demographic & Clinical Comparisons")
    
    # Create tabs for different comparison types
    tab1, tab2, tab3, tab4 = st.tabs(["👥 Demographics", "🏥 Clinical Factors", "📈 Summary Statistics", "📋 Data Quality"])
    
    with tab1:
        st.markdown("#### Demographic Characteristics Comparison")
        comparison_type = st.selectbox(
            "Select demographic variable:",
            ["Age Distribution", "Parity Distribution", "BMI Distribution", "Ethnicity Distribution"],
            key="demo_comparison"
        )
        
        if comparison_type == "Age Distribution":
            create_comparison_chart(df3, df4, 'Age_Group', 'Age Distribution: 3C vs 4th Degree Tears', 'Age Group')
        elif comparison_type == "Parity Distribution":
            create_comparison_chart(df3, df4, 'Parity_Group_v2', 'Parity Distribution: 3C vs 4th Degree Tears', 'Parity Group')
        elif comparison_type == "BMI Distribution":
            create_comparison_chart(df3, df4, 'BMI_Group', 'BMI Distribution: 3C vs 4th Degree Tears', 'BMI Category')
        elif comparison_type == "Ethnicity Distribution":
            create_comparison_chart(df3, df4, 'EthnicOrigin_cat', 'Ethnicity Distribution: 3C vs 4th Degree Tears', 'Ethnic Group')
    
    with tab2:
        st.markdown("#### Clinical Factors Comparison")
        clinical_comparison = st.selectbox(
            "Select clinical variable:",
            ["Repair Type Distribution", "Previous OASI History", "Treatment Patterns"],
            key="clinical_comparison"
        )
        
        if clinical_comparison == "Repair Type Distribution":
            create_comparison_chart(df3, df4, 'Repair_Type', 'Repair Type: 3C vs 4th Degree Tears', 'Repair Type')
        elif clinical_comparison == "Previous OASI History":
            # Previous OASI comparison
            prev_3c_rate = (df3['Prev_OASI_Flag'] == 'Yes').sum() / df3['Prev_OASI_Flag'].notna().sum() * 100
            prev_4th_rate = (df4['Prev_OASI_Flag'] == 'Yes').sum() / df4['Prev_OASI_Flag'].notna().sum() * 100
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("3C Tears", f"{prev_3c_rate:.1f}%", help="Rate of previous OASI")
            with col2:
                st.metric("4th Degree Tears", f"{prev_4th_rate:.1f}%", help="Rate of previous OASI")
                
        elif clinical_comparison == "Treatment Patterns":
            # Treatment comparison
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**3C Tears Treatment:**")
                lax_3c = (df3['Q12_Lax'] == 'Yes').sum() / len(df3) * 100
                abx_3c = (df3['Q13_Abx'] == 'Yes').sum() / len(df3) * 100
                st.write(f"- Laxatives: {lax_3c:.1f}%")
                st.write(f"- Antibiotics: {abx_3c:.1f}%")
            
            with col2:
                st.markdown("**4th Degree Tears Treatment:**")
                lax_4th = (df4['Q12_Lax'] == 'Yes').sum() / len(df4) * 100
                abx_4th = (df4['Q13_Abx'] == 'Yes').sum() / len(df4) * 100
                st.write(f"- Laxatives: {lax_4th:.1f}%")
                st.write(f"- Antibiotics: {abx_4th:.1f}%")
    
    with tab3:
        st.markdown("#### Summary Statistics Comparison")
        
        # Create comprehensive summary table
        summary_data = {
            'Characteristic': ['Sample Size', 'Mean Age (years)', 'Mean BMI (kg/m²)', 'Primigravida (%)', 'Overlapping Repair (%)'],
            '3C Tears': [
                len(df3),
                f"{df3['Age_at_deliveryDate'].mean():.1f}",
                f"{df3['bmi'].mean():.1f}",
                f"{(df3['Parity_Group_v2'] == '<1').mean()*100:.1f}%",
                f"{(df3['Repair_Type'] == 'Overlapping').mean()*100:.1f}%"
            ],
            '4th Degree Tears': [
                len(df4),
                f"{df4['Age_at_deliveryDate'].mean():.1f}",
                f"{df4['bmi'].mean():.1f}",
                f"{(df4['Parity_Group_v2'] == '<1').mean()*100:.1f}%",
                f"{(df4['Repair_Type'] == 'Overlapping').mean()*100:.1f}%"
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
        # Download summary
        summary_csv = summary_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Summary Comparison CSV",
            data=summary_csv,
            file_name="3c_vs_4th_degree_summary.csv",
            mime="text/csv",
            key="download_3c_4th_summary"
        )
    
    with tab4:
        st.markdown("#### Data Completeness Assessment")
        
        # Data quality metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**3C Tears Data Quality:**")
            st.write(f"- Total records: {len(df3)}")
            st.write(f"- Age missing: {df3['Age_at_deliveryDate'].isna().sum()} ({df3['Age_at_deliveryDate'].isna().mean()*100:.1f}%)")
            st.write(f"- BMI missing: {df3['bmi'].isna().sum()} ({df3['bmi'].isna().mean()*100:.1f}%)")
            st.write(f"- Mode of delivery missing: {(df3['MOD_ALL'] == 'MISSING').sum()} ({(df3['MOD_ALL'] == 'MISSING').mean()*100:.1f}%)")
        
        with col2:
            st.markdown("**4th Degree Tears Data Quality:**")
            st.write(f"- Total records: {len(df4)}")
            st.write(f"- Age missing: {df4['Age_at_deliveryDate'].isna().sum()} ({df4['Age_at_deliveryDate'].isna().mean()*100:.1f}%)")
            st.write(f"- BMI missing: {df4['bmi'].isna().sum()} ({df4['bmi'].isna().mean()*100:.1f}%)")
            st.write(f"- Mode of delivery missing: {(df4['MOD_ALL'] == 'MISSING').sum()} ({(df4['MOD_ALL'] == 'MISSING').mean()*100:.1f}%)")

def show_part3_repair_comparisons(df4):
    st.markdown('<h2 class="section-header">Part III: Questions D-F - Repair Type Comparisons</h2>', unsafe_allow_html=True)
    
    # Research questions with enhanced presentation
    st.info("""
    🎯 **Research Questions D, E, F:**
    
    **D.** Any significant difference between the rate of immediate and late bowel complications between tears repaired with end-to-end vs overlapping?
    
    **E.** Any significant difference between the rate of immediate and late urinary complications between tears repaired with end-to-end vs overlapping?
    
    **F.** Any significant difference between the rate of immediate and late vaginal complications between tears repaired with end-to-end vs overlapping?
    
    **Methodology:** Chi-squared tests and Fisher's exact tests for categorical comparisons between repair techniques.
    """)
    
    # Enhanced analysis with organized tabs
    st.markdown("### 🔬 Statistical Analysis Results")
    
    # Create tabs for each question group
    tab_d, tab_e, tab_f, tab_summary = st.tabs(["🔴 Question D: Bowel", "🔵 Question E: Urinary", "🟡 Question F: Vaginal", "📊 Summary Results"])
    
    # Filter to main repair types
    df_comp = df4[df4['Repair_Type'].isin(['Overlapping', 'End to end'])].copy()
    
    # Recreate comparison flags
    def create_any_comp_flag(df, columns):
        df_numeric = df[columns].apply(lambda x: x.str.strip().str.upper() == 'Y').astype(int)
        return (df_numeric.sum(axis=1) > 0).astype(int)
    
    # Define column groups
    bowel_imm = ['Bowel (immediate issues)_Faecal incontinence', 'Bowel (immediate issues)_Flatus incontinence', 'Bowel (immediate issues)_Urgency of stool']
    bowel_lt = ['Bowel (long term issues)_Faecal incontinence', 'Bowel (long term issues)_Flatus incontinence', 'Bowel (long term issues)_Urgency']
    urinary_imm = ['Urinary problems (immediate issues)_Urgency', 'Urinary problems (immediate issues)_Frequency', 'Urinary problems (immediate issues)_Leakage', 'Urinary problems (immediate issues)_Leakage on strenuous activity', 'Urinary problems (immediate issues)_Voiding dysfunction']
    urinary_lt = ['Urinary problems (long term issues)_Urgency', 'Urinary problems (long term issues)_Frequency', 'Urinary problems (long term issues)_Leakage', 'Urinary problems (long term issues)_Leakage on strenuous activity', 'Urinary problems (long term issues)_Voiding dysfunction']
    vaginal_imm = ['Vaginal problems (immediate)_Body image', 'Vaginal problems (immediate)_Dyspareunia', 'Vaginal problems (immediate)_Vaginal lump']
    vaginal_lt = ['Vaginal problems (long term)_Body image', 'Vaginal problems (long term)_Dyspareunia', 'Vaginal problems (long term)_Vaginal lump']
    
    # Create flags
    df_comp['Bowel_imm_defer_comp'] = (df_comp['Bowel (immediate issues)_Able to defer bowels?'].str.strip().str.upper() == 'N').astype(int)
    df_comp['Bowel_lt_defer_comp'] = (df_comp['Bowel (long term issues)_Able to defer bowels?'].str.strip().str.upper() == 'N').astype(int)
    
    df_comp['Any_Bowel_Immediate'] = ((create_any_comp_flag(df_comp, bowel_imm) + df_comp['Bowel_imm_defer_comp']) > 0).astype(int)
    df_comp['Any_Bowel_Late'] = ((create_any_comp_flag(df_comp, bowel_lt) + df_comp['Bowel_lt_defer_comp']) > 0).astype(int)
    df_comp['Any_Urinary_Immediate'] = create_any_comp_flag(df_comp, urinary_imm)
    df_comp['Any_Urinary_Late'] = create_any_comp_flag(df_comp, urinary_lt)
    df_comp['Any_Vaginal_Immediate'] = create_any_comp_flag(df_comp, vaginal_imm)
    df_comp['Any_Vaginal_Late'] = create_any_comp_flag(df_comp, vaginal_lt)
    
    with tab_d:
        st.markdown("#### Question D: Bowel Complications by Repair Type")
        show_repair_comparison_analysis(df_comp, "Bowel", ["Any_Bowel_Immediate", "Any_Bowel_Late"])
    
    with tab_e:
        st.markdown("#### Question E: Urinary Complications by Repair Type")
        show_repair_comparison_analysis(df_comp, "Urinary", ["Any_Urinary_Immediate", "Any_Urinary_Late"])
    
    with tab_f:
        st.markdown("#### Question F: Vaginal Complications by Repair Type")
        show_repair_comparison_analysis(df_comp, "Vaginal", ["Any_Vaginal_Immediate", "Any_Vaginal_Late"])
    
    with tab_summary:
        st.markdown("#### 📊 Complete Statistical Summary")
        show_complete_repair_summary(df_comp)

def show_repair_comparison_analysis(df_comp, complication_type, comp_vars):
    """Show detailed analysis for one complication type."""
    
    st.markdown(f"**Analysis Focus:** {complication_type} complications comparing End-to-end vs Overlapping repair")
    
    # Create side-by-side comparison
    col1, col2 = st.columns(2)
    
    for i, comp_var in enumerate(comp_vars):
        period = "Immediate" if "Immediate" in comp_var else "Long-term"
        
        with col1 if i == 0 else col2:
            st.markdown(f"##### {period} {complication_type} Issues")
            
            if comp_var in df_comp.columns:
                # Calculate rates by repair type
                rates_by_repair = df_comp.groupby('Repair_Type')[comp_var].agg(['count', 'sum', 'mean']).reset_index()
                rates_by_repair['rate'] = rates_by_repair['mean'] * 100
                
                # Display metrics
                overlapping_rate = rates_by_repair[rates_by_repair['Repair_Type'] == 'Overlapping']['rate'].iloc[0]
                endtoend_rate = rates_by_repair[rates_by_repair['Repair_Type'] == 'End to end']['rate'].iloc[0]
                
                st.metric("Overlapping", f"{overlapping_rate:.1f}%")
                st.metric("End-to-end", f"{endtoend_rate:.1f}%")
                
                # Statistical test
                contingency = pd.crosstab(df_comp['Repair_Type'], df_comp[comp_var])
                if contingency.shape == (2, 2):
                    chi2_stat, chi2_p, _, expected = chi2_contingency(contingency)
                    
                    if (expected < 5).any():
                        _, fisher_p = fisher_exact(contingency)
                        test_used = "Fisher's Exact"
                        p_value = fisher_p
                    else:
                        test_used = "Chi-squared"
                        p_value = chi2_p
                    
                    # Display test results
                    if p_value < 0.05:
                        st.success(f"**Significant difference** (p = {p_value:.4f})")
                    else:
                        st.info(f"**No significant difference** (p = {p_value:.4f})")
                    
                    st.caption(f"Test used: {test_used}")
    
    # Visualization for this complication type
    if len(comp_vars) == 2 and all(var in df_comp.columns for var in comp_vars):
        create_repair_comparison_chart(df_comp, comp_vars, complication_type)

def show_complete_repair_summary(df_comp):
    """Show complete summary of all repair type comparisons."""
    
    st.markdown("**Complete Statistical Results: End-to-end vs Overlapping Repair**")
    
    # Define all comparisons
    comparisons = {
        "D1: Immediate Bowel": "Any_Bowel_Immediate",
        "D2: Long-term Bowel": "Any_Bowel_Late",
        "E1: Immediate Urinary": "Any_Urinary_Immediate", 
        "E2: Long-term Urinary": "Any_Urinary_Late",
        "F1: Immediate Vaginal": "Any_Vaginal_Immediate",
        "F2: Long-term Vaginal": "Any_Vaginal_Late"
    }
    
    results_data = []
    
    for comp_name, comp_var in comparisons.items():
        if comp_var in df_comp.columns:
            # Create contingency table
            contingency = pd.crosstab(df_comp['Repair_Type'], df_comp[comp_var])
            
            if contingency.shape == (2, 2):
                # Run statistical test
                chi2_stat, chi2_p, _, expected = chi2_contingency(contingency)
                
                if (expected < 5).any():
                    odds_ratio, fisher_p = fisher_exact(contingency)
                    test_used = "Fisher's Exact"
                    p_value = fisher_p
                else:
                    test_used = "Chi-squared"
                    p_value = chi2_p
                
                # Calculate rates
                overlapping_rate = contingency.loc['Overlapping', 1] / contingency.loc['Overlapping'].sum() * 100
                endtoend_rate = contingency.loc['End to end', 1] / contingency.loc['End to end'].sum() * 100
                
                results_data.append({
                    'Comparison': comp_name,
                    'Overlapping Rate': f"{overlapping_rate:.1f}%",
                    'End-to-end Rate': f"{endtoend_rate:.1f}%",
                    'Test Used': test_used,
                    'p-value': f"{p_value:.4f}",
                    'Significant': "✓" if p_value < 0.05 else "✗",
                    'Effect Size': f"{abs(overlapping_rate - endtoend_rate):.1f}% difference"
                })
    
    # Display comprehensive results table
    if results_data:
        results_df = pd.DataFrame(results_data)
        
        # Enhanced styling for summary table
        def style_summary_table(row):
            if row['Significant'] == '✓':
                return ['background: linear-gradient(90deg, #d4edda, #c3e6cb); color: #155724; font-weight: 600'] * len(row)
            return ['background: #f8f9fa; color: #2c3e50'] * len(row)
        
        styled_results = results_df.style.apply(style_summary_table, axis=1)
        styled_results = styled_results.set_table_styles([
            {'selector': 'th', 'props': [('background-color', '#34495e'), ('color', 'white'), ('font-weight', 'bold'), ('padding', '12px')]},
            {'selector': 'td', 'props': [('padding', '10px'), ('border-bottom', '1px solid #dee2e6')]}
        ])
        
        st.dataframe(styled_results, use_container_width=True)
        
        # Key findings summary
        significant_count = (results_df['Significant'] == '✓').sum()
        
        if significant_count > 0:
            st.success(f"🎯 **{significant_count}** out of **{len(results_df)}** comparisons showed statistically significant differences (p < 0.05)")
            
            # Show significant findings
            sig_results = results_df[results_df['Significant'] == '✓']
            if not sig_results.empty:
                st.markdown("**Significant Findings:**")
                for _, row in sig_results.iterrows():
                    st.write(f"• **{row['Comparison']}**: {row['Effect Size']} (p = {row['p-value']})")
        else:
            st.info(f"📊 **No significant differences** found between repair types across all {len(results_df)} comparisons (all p ≥ 0.05)")
        
        # Download comprehensive results
        comp_csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Complete Repair Comparison Results CSV",
            data=comp_csv,
            file_name="complete_repair_type_comparisons.csv",
            mime="text/csv",
            key="download_complete_repair_stats"
        )

def create_repair_comparison_chart(df_comp, comp_vars, complication_type):
    """Create visualization for repair type comparison."""
    
    # Prepare data for visualization
    chart_data = []
    
    for comp_var in comp_vars:
        period = "Immediate" if "Immediate" in comp_var else "Long-term"
        rates_by_repair = df_comp.groupby('Repair_Type')[comp_var].mean() * 100
        
        for repair_type in ['Overlapping', 'End to end']:
            if repair_type in rates_by_repair.index:
                chart_data.append({
                    'Repair Type': repair_type,
                    'Period': period,
                    'Complication Rate (%)': rates_by_repair[repair_type]
                })
    
    if chart_data:
        chart_df = pd.DataFrame(chart_data)
        
        fig = px.bar(
            chart_df,
            x='Period',
            y='Complication Rate (%)',
            color='Repair Type',
            title=f'{complication_type} Complications: Immediate vs Long-term by Repair Type',
            barmode='group',
            color_discrete_map={'Overlapping': '#3498db', 'End to end': '#e74c3c'}
        )
        
        fig.update_layout(
            height=400,
            font=dict(family='Inter', size=12),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def display_regression_results(model_results, title):
    """Display enhanced regression results with better formatting."""
    
    st.markdown(f"### 📊 {title} - Regression Analysis")
    
    # Model performance metrics in a nice layout
    if not model_results.get('regularized', False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Sample Size", f"{model_results['n_obs']}")
        with col2:
            st.metric("Pseudo R²", f"{model_results['pseudo_r2']:.3f}", 
                     help="Higher values indicate better model fit (0-1 scale)")
        with col3:
            llr_p = model_results['llr_p']
            st.metric("Overall p-value", f"{llr_p:.4f}", 
                     delta="Significant" if llr_p < 0.05 else "Not significant")
        with col4:
            # Model quality indicator
            if model_results['pseudo_r2'] > 0.2:
                quality = "Good"
                quality_color = "normal"
            elif model_results['pseudo_r2'] > 0.1:
                quality = "Moderate" 
                quality_color = "normal"
            else:
                quality = "Limited"
                quality_color = "inverse"
            st.metric("Model Fit", quality, help="Based on Pseudo R² value")
    else:
        st.info("📋 Regularized model used due to data limitations - standard fit metrics not available")
        st.metric("Sample Size", f"{model_results['n_obs']}")
    
    # Academic-style results presentation
    st.markdown('<div class="academic-card">', unsafe_allow_html=True)
    st.markdown("#### 🎯 Multivariable Logistic Regression Results")
    
    or_df = model_results['odds_ratios'].copy()
    
    # Enhanced table with academic formatting
    if 'Significance' in or_df.columns:
        # Create a more sophisticated styling
        def style_academic_table(row):
            if row['Significance'] == '✓':
                return ['background: linear-gradient(90deg, #d4edda, #c3e6cb); color: #155724; font-weight: 600'] * len(row)
            return ['background: #f8f9fa; color: #2c3e50'] * len(row)
        
        styled_df = or_df.style.apply(style_academic_table, axis=1)
        styled_df = styled_df.set_table_styles([
            {'selector': 'th', 'props': [('background-color', '#34495e'), ('color', 'white'), ('font-weight', 'bold')]},
            {'selector': 'td', 'props': [('padding', '12px'), ('border-bottom', '1px solid #dee2e6')]}
        ])
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Academic findings summary
        significant_factors = or_df[or_df['Significance'] == '✓']
        if not significant_factors.empty:
            st.markdown("""
            <div class="finding-box">
                <h4>📊 Significant Associations Identified</h4>
            """, unsafe_allow_html=True)
            
            for _, row in significant_factors.iterrows():
                or_val = float(row['OR (95% CI)'].split('(')[0].strip())
                interpretation = "increased risk" if or_val > 1 else "protective effect"
                st.markdown(f"<p><strong>{row['Predictor']}</strong>: {row['OR (95% CI)']} - {interpretation}</p>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="interpretation-box">
                <h4>📋 Statistical Interpretation</h4>
                <p>No individual predictors reached statistical significance (p < 0.05) in this multivariable model. 
                This suggests that complications may be influenced by factors not captured in this analysis or 
                may occur relatively randomly across the measured characteristics.</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.dataframe(or_df, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Download button for this analysis
    or_csv = or_df.to_csv(index=False)
    st.download_button(
        label=f"📥 Download {title} Results CSV",
        data=or_csv,
        file_name=f"{title.replace(' ', '_').replace('-', '_').lower()}_results.csv",
        mime="text/csv",
        key=f"download_results_{title.replace(' ', '_').replace('-', '_').lower()}"
    )
    
    # Forest plot with enhanced styling
    st.markdown("#### 🌲 Forest Plot Visualization")
    create_forest_plot(model_results)
    
    # Model interpretation guide
    with st.expander("📚 How to interpret these results"):
        st.markdown("""
        **Odds Ratio (OR) Interpretation:**
        - **OR = 1.0**: No association with the outcome
        - **OR > 1.0**: Increased risk (risk factor)
        - **OR < 1.0**: Decreased risk (protective factor)
        
        **Statistical Significance:**
        - **p < 0.05**: Statistically significant association
        - **95% CI**: If the confidence interval crosses 1.0, the result is not significant
        
        **Model Quality:**
        - **Pseudo R²**: Proportion of variance explained (higher = better fit)
        - **Overall p-value**: Tests if the model is better than chance
        """)
    
    # Add model diagnostics
    if not model_results.get('regularized', False):
        with st.expander("🔍 Model Diagnostics"):
            st.markdown(f"""
            **Model Statistics:**
            - Sample size: {model_results['n_obs']} patients
            - Pseudo R²: {model_results['pseudo_r2']:.3f}
            - Overall model p-value: {model_results['llr_p']:.4f}
            
            **Model Interpretation:**
            - This model explains {model_results['pseudo_r2']*100:.1f}% of the variance in {title.lower()}
            - The overall model is {'significant' if model_results['llr_p'] < 0.05 else 'not significant'} (p = {model_results['llr_p']:.4f})
            """)
    else:
        with st.expander("ℹ️ Regularized Model Information"):
            st.markdown("""
            **Why was regularization used?**
            - The standard logistic regression failed due to data sparsity or perfect separation
            - Regularization helps stabilize the model when there are few cases in some categories
            - Results should be interpreted more cautiously as confidence intervals are not available
            """)

if __name__ == "__main__":
    main()
