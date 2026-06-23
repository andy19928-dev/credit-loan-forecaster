import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import math

# --- 產品大項對齊對應工具 ---
def map_product_category(prod):
    prod_str = str(prod).strip()
    if '大口' in prod_str:
        return '大口貸'
    elif '小口' in prod_str:
        return '小口貸'
    elif '備用' in prod_str or '備' in prod_str:
        return '備用金'
    return '其他'

# --- 智慧型欄位模糊對齊工具 (防重複升級版) ---
def fuzzy_rename_columns(df):
    # 先去除所有欄位名稱的前後空白
    df.columns = df.columns.astype(str).str.strip()
    
    rename_dict = {}
    assigned_targets = set() # 記錄已經成功對齊的標準欄位，避免重複
    
    for c in df.columns:
        target = None
        if '進件日' in c: target = '進件日'
        elif '動撥日' in c: target = '撥貸_動撥日'
        elif '是否核准' in c: target = '審核_是否核准'
        elif '產品類型' in c or '產品' in c: target = '明細_產品類型名稱'
        elif '資金用途' in c: target = '申請書_資金用途'
        elif '行業別_DESC' in c or ('行業別' in c and '代碼' not in c): target = '申書_現任公司行業別_DESC'
        elif '申貸金額' in c: target = '申請書_申貸金額'
        elif '核准信貸額度' in c or '核准額度' in c: target = '審核_核准信貸額度'
        elif '撥貸金額' in c: target = '撥貸_撥貸金額'
        elif '是否撥貸' in c: target = '撥貸_是否撥貸'
        elif '申請人年收' in c or '年收入' in c or '年收' in c: target = '申書_申請人年收入'
        
        # 如果找到目標，且該目標還沒被其他欄位佔用，才進行改名
        if target and target not in assigned_targets:
            rename_dict[c] = target
            assigned_targets.add(target)
            
    df.rename(columns=rename_dict, inplace=True)
    # 終極防線：自動剔除任何名稱重複的欄位 (保留第一個遇到的)
    df = df.loc[:, ~df.columns.duplicated()]
    
    # 建立與對齊產品大類（大口貸、小口貸、備用金、其他）
    if '明細_產品類型名稱' in df.columns:
        df['產品大類'] = df['明細_產品類型名稱'].apply(map_product_category)
    else:
        df['產品大類'] = '其他'
        df['明細_產品類型名稱'] = '其他'
        
    return df

# --- 核心預測與優化引擎 ---
class CreditLoanForecastingPlatform:
    def __init__(self):
        self.approval_matrix = {}
        self.drawdown_matrix = {}
        self.time_lag_matrix = {}
        self.avg_amount_matrix = {}
        self.strategy_matrix = {} # 用於尋找最佳達標策略
        self.global_approval_rate = 0.5
        # 儲存分箱邊界，讓未來資料可以對齊相同的分級
        self.income_bins = []
        self.loan_amt_bins = []
        
    def preprocess_historical_data(self, df):
        df['進件日'] = pd.to_datetime(df['進件日'], errors='coerce')
        if '撥貸_動撥日' in df.columns:
            df['撥貸_動撥日'] = pd.to_datetime(df['撥貸_動撥日'], errors='coerce')
        else:
            df['撥貸_動撥日'] = pd.NaT
            
        amount_cols = ['申請書_申貸金額', '審核_核准信貸額度', '撥貸_撥貸金額', '申書_申請人年收入']
        for col in amount_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '').str.strip()
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                
        if '審核_是否核准' in df.columns and '撥貸_動撥日' in df.columns:
            approved_mask = (df['審核_是否核准'] == 'Y') & (df['撥貸_動撥日'].notna())
            df.loc[approved_mask, '遞延天數'] = (df['撥貸_動撥日'] - df['進件日']).dt.days
        return df

    def train_forecasting_engines(self, df):
        # 確保訓練集有產品大類欄位
        if '產品大類' not in df.columns:
            df['產品大類'] = df['明細_產品類型名稱'].apply(map_product_category)

        # 訓練時動態計算分箱邊界 (使用 retbins=True)
        _, self.income_bins = pd.qcut(df['申書_申請人年收入'].clip(lower=1), q=4, retbins=True, duplicates='drop')
        _, self.loan_amt_bins = pd.qcut(df['申請書_申貸金額'].clip(lower=1), q=4, retbins=True, duplicates='drop')
        
        # 為了容許新資料超出邊界，將最外層邊界設為無限大
        self.income_bins[0], self.income_bins[-1] = -np.inf, np.inf
        self.loan_amt_bins[0], self.loan_amt_bins[-1] = -np.inf, np.inf
        
        df['年收入級距'] = pd.cut(df['申書_申請人年收入'].clip(lower=1), bins=self.income_bins, labels=['低', '中偏低', '中偏高', '高'], duplicates='drop')
        df['申貸金額級距'] = pd.cut(df['申請書_申貸金額'].clip(lower=1), bins=self.loan_amt_bins, labels=['小額', '中額', '大額', '高額'], duplicates='drop')
        
        # 訓練核准率基準
        approval_stats = df.groupby([
            '產品大類', '申請書_資金用途', '申書_現任公司行業別_DESC', '年收入級距', '申貸金額級距'
        ], observed=True)['審核_是否核准'].value_counts().unstack().fillna(0)
        
        if 'Y' not in approval_stats.columns: approval_stats['Y'] = 0
        if 'N' not in approval_stats.columns: approval_stats['N'] = 0
        self.approval_matrix = (approval_stats['Y'] / (approval_stats['Y'] + approval_stats['N'])).to_dict()
        self.global_approval_rate = (df['審核_是否核准'] == 'Y').mean()

        # 訓練撥貸率與金額
        approved_df = df[df['審核_是否核准'] == 'Y'].copy()
        if len(approved_df) > 0:
            self.drawdown_matrix = approved_df.groupby(['產品大類', '年收入級距', '申貸金額級距'], observed=True)['撥貸_是否撥貸'].apply(lambda x: (x == 'Y').mean()).to_dict()
            self.avg_amount_matrix = approved_df[approved_df['撥貸_是否撥貸'] == 'Y'].groupby(['產品大類', '年收入級距', '申貸金額級距'], observed=True)['撥貸_撥貸金額'].mean().to_dict()
            self.time_lag_matrix = approved_df[approved_df['撥貸_是否撥貸'] == 'Y'].groupby(['產品大類'], observed=True)['遞延天數'].mean().fillna(3).to_dict()

        # --- 計算策略最佳化矩陣 (計算每種組合的「期望值 EV」) ---
        self.strategy_matrix = {}
        for (prod, purpose, ind), group in df.groupby(['產品大類', '申請書_資金用途', '申書_現任公司行業別_DESC'], observed=True):
            app_rate = (group['審核_是否核准'] == 'Y').mean()
            app_group = group[group['審核_是否核准'] == 'Y']
            if len(app_group) > 0:
                draw_rate = (app_group['撥貸_是否撥貸'] == 'Y').mean()
                draw_amt = app_group[app_group['撥貸_是否撥貸'] == 'Y']['撥貸_撥貸金額'].mean()
                if pd.isna(draw_amt): draw_amt = 0
            else:
                draw_rate, draw_amt = 0, 0
                
            ev_per_app = app_rate * draw_rate * draw_amt
            if ev_per_app > 0:
                if prod not in self.strategy_matrix:
                    self.strategy_matrix[prod] = []
                self.strategy_matrix[prod].append({
                    '資金用途': purpose, 
                    '行業別': ind, 
                    '核准率': app_rate,
                    '撥貸率': draw_rate,
                    '件均撥貸額': draw_amt,
                    '每件期望撥款量': ev_per_app
                })
        
        # 依照期望值排序，方便後續給出最佳建議
        for prod in self.strategy_matrix:
            self.strategy_matrix[prod] = sorted(self.strategy_matrix[prod], key=lambda x: x['每件期望撥款量'], reverse=True)

    def generate_row_level_predictions(self, seed_df, start_date, end_date):
        # 1. 根據前60天資料，模擬出未來區間的每日進件量
        seed_df['進件日'] = pd.to_datetime(seed_df['進件日'], errors='coerce')
        seed_days = (seed_df['進件日'].max() - seed_df['進件日'].min()).days
        seed_days = max(seed_days, 1) # 防呆，避免除以0
        
        target_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1
        daily_intake_rate = len(seed_df) / seed_days
        target_total_apps = int(daily_intake_rate * target_days)
        
        # 隨機抽樣擴展或壓縮資料以符合未來天數
        sim_df = seed_df.sample(n=target_total_apps, replace=True).reset_index(drop=True)
        
        # 隨機分派未來區間的進件日
        random_days_offset = np.random.randint(0, target_days, size=target_total_apps)
        sim_df['進件日'] = pd.to_datetime(start_date) + pd.to_timedelta(random_days_offset, unit='d')
        
        # 動態分箱 (解決舊版硬編碼問題)
        if len(self.income_bins) > 0 and len(self.loan_amt_bins) > 0:
            sim_df['年收入級距'] = pd.cut(sim_df.get('申書_申請人年收入', 0).clip(lower=1), bins=self.income_bins, labels=['低', '中偏低', '中偏高', '高'], duplicates='drop')
            sim_df['申貸金額級距'] = pd.cut(sim_df.get('申請書_申貸金額', 0).clip(lower=1), bins=self.loan_amt_bins, labels=['小額', '中額', '大額', '高額'], duplicates='drop')
        else:
            sim_df['年收入級距'] = '中偏高'
            sim_df['申貸金額級距'] = '中額'

        results = []
        for idx, row in sim_df.iterrows():
            prod = row.get('產品大類', '其他')
            original_prod = row.get('明細_產品類型名稱', '未知產品')
            purpose = row.get('申請書_資金用途', '其他')
            industry = row.get('申書_現任公司行業別_DESC', '其他')
            income_grp = row.get('年收入級距', '中偏高')
            amt_grp = row.get('申貸金額級距', '中額')
            
            # 特徵比對預測
            features = (prod, purpose, industry, income_grp, amt_grp)
            p_approve = self.approval_matrix.get(features, self.global_approval_rate)
            
            drawdown_features = (prod, income_grp, amt_grp)
            p_drawdown = self.drawdown_matrix.get(drawdown_features, 0.5)
            pred_amount = self.avg_amount_matrix.get(drawdown_features, 150000)
            pred_lag_days = int(np.round(self.time_lag_matrix.get(prod, 3)))
            
            pred_drawdown_date = row['進件日'] + timedelta(days=pred_lag_days)
            
            results.append({
                '明細_產品類型名稱': original_prod,
                '產品大類': prod,
                '申請書_資金用途': purpose,
                '申書_現任公司行業別_DESC': industry,
                '進件日': row['進件日'],
                # 使用 W-SUN 確保每週第一天是星期一
                '進件週別': row['進件日'].to_period('W-SUN').start_time,
                '預估核准數': 1 * p_approve,
                '預估未核准數': 1 * (1 - p_approve),
                '預估撥貸件數': 1 * p_approve * p_drawdown,
                '預估撥款金額': pred_amount * p_approve * p_drawdown,
                '預估撥款日期': pred_drawdown_date,
                '預估撥款週別': pred_drawdown_date.to_period('W-SUN').start_time
            })
            
        return pd.DataFrame(results)

    def aggregate_weekly_report(self, df):
        if df.empty:
            return pd.DataFrame()
            
        incoming_group = df.groupby('進件週別').agg({'進件日': 'count', '預估核准數': 'sum', '預估未核准數': 'sum'}).rename(columns={'進件日': '預估進件數'})
        incoming_group['預估核准率'] = incoming_group['預估核准數'] / (incoming_group['預估核准數'] + incoming_group['預估未核准數'])
        
        drawdown_group = df.groupby('預估撥款週別').agg({'預估撥貸件數': 'sum', '預估撥款金額': 'sum'}).rename(columns={'預估撥款金額': '預估新撥量'})
        
        final_report = pd.merge(incoming_group, drawdown_group, left_index=True, right_index=True, how='outer').fillna(0)
        final_report['預估撥貸率'] = np.where(final_report['預估核准數'] > 0, final_report['預估撥貸件數'] / final_report['預估核准數'], 0)
        
        # 整理外觀 (日期轉字串，星期一)
        final_report.index = final_report.index.strftime('%Y-%m-%d') + ' (週一)'
        final_report.index.name = '週別 (起始日)'
        
        # 格式化輸出前保留數值供圖表使用
        final_report['_核准率數值'] = final_report['預估核准率']
        final_report['_撥貸率數值'] = final_report['預估撥貸率']
        final_report['_新撥量數值'] = final_report['預估新撥量']
        
        final_report['預估核准率'] = (final_report['預估核准率'] * 100).round(1).astype(str) + '%'
        final_report['預估撥貸率'] = (final_report['預估撥貸率'] * 100).round(1).astype(str) + '%'
        final_report['預估新撥量'] = final_report['預估新撥量'].round(0).apply(lambda x: f"${x:,.0f}")
        
        return final_report[['預估進件數', '預估核准率', '預估撥貸率', '預估新撥量', '_新撥量數值']]

# --- 網頁前端介面 (Streamlit) ---
st.set_page_config(page_title="信貸業績預估平台", layout="wide")
st.title("📊 信貸業績預估與智慧達標平台")

if 'platform' not in st.session_state:
    st.session_state.platform = CreditLoanForecastingPlatform()

# 區塊 1：歷史資料訓練
with st.expander("🛠️ 第一步：上傳歷史進件明細 (訓練預測模型)", expanded=True):
    hist_file = st.file_uploader("請上傳歷史已完結案件資料 (以計算核准與撥貸特徵)", type=['csv'], key='hist')
    if hist_file is not None:
        hist_df = pd.read_csv(hist_file, skiprows=2, encoding='cp950', encoding_errors='ignore', on_bad_lines='skip')
        hist_df = fuzzy_rename_columns(hist_df)
        with st.spinner('正在訓練客群特徵矩陣與最佳化決策樹...'):
            cleaned_df = st.session_state.platform.preprocess_historical_data(hist_df)
            st.session_state.platform.train_forecasting_engines(cleaned_df)
        st.success("✅ 模型訓練完成！已成功對齊大項產品，並抓取各項行業與資金用途之獲利期望值。")

st.divider()

# 區塊 2：未來推估設定與目標配置
st.header("🎯 第二步：設定未來預估區間與業績目標")
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("推估起始日期", pd.Timestamp.today())
with col2:
    end_date = st.date_input("推估結束日期", pd.Timestamp.today() + timedelta(days=30))

st.info("💡 提示：請上傳「過去 60 天」的近期進件資料。模型將依據這 60 天的進件水準與客群輪廓，推演至您設定的未來日期區間。")
future_file = st.file_uploader("請上傳作為推估基準的近期 60 天資料", type=['csv'], key="future")

if future_file is not None and len(st.session_state.platform.approval_matrix) > 0:
    seed_df = pd.read_csv(future_file, skiprows=2, encoding='cp950', encoding_errors='ignore', on_bad_lines='skip')
    seed_df = fuzzy_rename_columns(seed_df)
    seed_df = st.session_state.platform.preprocess_historical_data(seed_df)
    
    # 鎖定產品大項清單為大口貸、小口貸、備用金
    target_products = ["大口貸", "小口貸", "備用金"]
    
    st.subheader("🏆 設定預估區間業績目標")
    targets = {}
    
    # 固定為 3 個大項產生輸入框
    target_cols = st.columns(3)
    for i, prod in enumerate(target_products):
        with target_cols[i]:
            st.markdown(f"**產品大項：{prod}**")
            t_apps = st.number_input(f"目標進件數", min_value=0, value=0, key=f"app_{prod}")
            t_amt = st.number_input(f"目標新撥量 (元)", min_value=0, value=0, step=100000, key=f"amt_{prod}")
            targets[prod] = {'進件': t_apps, '新撥': t_amt}
            st.write("---")

    if st.button("🚀 產出預估報告與策略建議", type="primary"):
        with st.spinner('正在生成逐筆模擬資料與最佳化路徑...'):
            # 產生行級別的預測基礎資料
            full_sim_df = st.session_state.platform.generate_row_level_predictions(seed_df, start_date, end_date)
            st.session_state['full_sim_df'] = full_sim_df
            st.session_state['targets'] = targets

st.divider()

# 區塊 3：多維度篩選與互動報表
if 'full_sim_df' in st.session_state:
    st.header("📈 第三步：預估報表與智慧建議")
    
    sim_df = st.session_state['full_sim_df']
    targets = st.session_state['targets']
    
    # --- 篩選器 ---
    st.markdown("### 🔍 多維度交叉篩選")
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        # 篩選選單直接提供對齊後的產品大類，並預設勾選主力三大項
        available_categories = sim_df['產品大類'].unique()
        default_selections = [p for p in available_categories if p in ["大口貸", "小口貸", "備用金"]]
        sel_prod = st.multiselect("產品大類", options=available_categories, default=default_selections if default_selections else available_categories)
    with filter_col2:
        sel_purpose = st.multiselect("資金需求", options=sim_df['申請書_資金用途'].unique(), default=sim_df['申請書_資金用途'].unique())
    with filter_col3:
        sel_industry = st.multiselect("行業別", options=sim_df['申書_現任公司行業別_DESC'].unique(), default=sim_df['申書_現任公司行業別_DESC'].unique())
    
    # 套用篩選（使用產品大類作為篩選準則）
    mask = (
        sim_df['產品大類'].isin(sel_prod) & 
        sim_df['申請書_資金用途'].isin(sel_purpose) & 
        sim_df['申書_現任公司行業別_DESC'].isin(sel_industry)
    )
    filtered_df = sim_df[mask]
    
    # 產出週報表
    weekly_report = st.session_state.platform.aggregate_weekly_report(filtered_df)
    
    if not weekly_report.empty:
        st.dataframe(weekly_report.drop(columns=['_新撥量數值']), use_container_width=True)
        
        # --- 目標檢核與 AI 建議引擎 ---
        st.subheader("💡 智慧達標優化建議")
        
        has_advice = False
        for prod in ["大口貸", "小口貸", "備用金"]: # 嚴格檢驗三個核心大項
            if prod not in targets: continue
            
            # 統計當前大項產品的預估績效
            prod_df = filtered_df[filtered_df['產品大類'] == prod]
            actual_amt = prod_df['預估撥款金額'].sum()
            actual_apps = len(prod_df)
            
            target_amt = targets[prod]['新撥']
            
            if target_amt > 0 and actual_amt < target_amt:
                has_advice = True
                shortfall = target_amt - actual_amt
                
                # 尋找該產品大類的最佳推廣客群
                best_strategies = st.session_state.platform.strategy_matrix.get(prod, [])
                
                st.error(f"⚠️ **【{prod}】預估新撥量為 ${actual_amt:,.0f}，距新撥目標 ${target_amt:,.0f} 還差 ${shortfall:,.0f}**")
                
                if best_strategies:
                    best = best_strategies[0] # 取獲利期望值最高的組合
                    ev = best['每件期望撥款量']
                    if ev > 0:
                        needed_apps = math.ceil(shortfall / ev)
                        st.success(
                            f"✨ **【{prod}】最佳化達標建議：**\n\n"
                            f"建議行銷資源優先鎖定 **資金需求為「{best['資金用途']}」且 行業別為「{best['行業別']}」** 的客群。\n\n"
                            f"📈 歷史數據顯示，該客群核准率為 {best['核准率']*100:.1f}%、撥貸率為 {best['撥貸率']*100:.1f}%，平均每進件 1 案可貢獻 **${ev:,.0f}** 的新撥量。\n"
                            f"🎯 只要在預估期間內**額外引流約 {needed_apps} 件**該屬性的進件，即可用最少的新增進件數達到新撥目標！"
                        )
                else:
                    st.warning("目前歷史數據庫中，無該產品大類之充足轉換特徵，無法提供高精準度的客群推薦。")
            elif target_amt > 0:
                st.info(f"🎉 **【{prod}】預計可達標！預估新撥量 (${actual_amt:,.0f}) 已大於新撥目標 (${target_amt:,.0f})。**")
                
        if not has_advice:
            st.write("目前尚未設定目標，或設定之各核心大項產品均已順利達標！")
    else:
        st.warning("當前篩選條件下無預估資料。")