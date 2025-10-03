evaluation = f"This model scored {score+6}/12. "
    if score >= 2:
        evaluation += "The strong technical score suggests the stock is in oversold territory or showing strong momentum characteristics that typically precede upward moves. Technical buying signals are flashing, indicating favorable risk-reward for entries. These setups often attract momentum traders and can create self-fulfilling rallies."
    elif score >= 0:
        evaluation += "The neutral technical score indicates the stock is in balanced territory without extreme readings in either direction. Technical indicators are not providing strong directional signals, suggesting a wait-and-see approach may be prudent until clearer patterns emerge."
    else:
        evaluation += "The negative technical score warns of overbought conditions or deteriorating momentum that often precedes pullbacks. Technical indicators are flashing caution signals. Experienced traders typically avoid buying into overbought conditions, preferring to wait for healthier technical setups."
    
    details.insert(0, methodology)
    details.append(evaluation)
    
    return score, details

@st.cache_data(ttl=300)
def get_stock_data(ticker, period):
    try:
        stock = yf.Ticker(ticker)
        return stock.history(period=period), stock.info, None
    except Exception as e:
        return None, None, str(e)

# PAGES
if st.session_state.page == 'analysis':
    st.markdown("<div class='brbas-header'><h1 class='brbas-title' style='color: #ffffff !important;'>BARBAS</h1></div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([4, 2, 2])
    with col1:
        ticker = search_ticker(st.text_input("", "AAPL", placeholder="Ticker or company", label_visibility="collapsed"))
    with col2:
        period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=3, label_visibility="collapsed")
    with col3:
        depth = st.selectbox("Depth", ["Standard", "Detailed"], index=0, label_visibility="collapsed")
    
    with st.spinner(f"Analyzing {ticker}..."):
        data, info, error = get_stock_data(ticker, period)
    
    if error or data is None or data.empty:
        st.error("Data unavailable")
        st.stop()
    
    for ma in [20, 50, 100, 200]:
        data[f'MA{ma}'] = data['Close'].rolling(window=ma).mean()
    data = calculate_ema(data)
    data['RSI'] = calculate_rsi(data)
    data['MACD'], data['Signal'], data['Histogram'] = calculate_macd(data)
    
    val_score, val_details = analyze_valuation(info)
    mom_score, mom_details = analyze_momentum(data)
    earn_score, earn_details = analyze_earnings(info)
    tech_score, tech_details = analyze_technical(data)
    
    confidence = calculate_confidence_score(info, data, val_score, mom_score, earn_score, tech_score)
    rec, rec_class = get_recommendation_from_confidence(confidence)
    
    if ticker not in st.session_state.portfolio:
        if st.button("Add to Portfolio"):
            st.session_state.portfolio.append(ticker)
            st.success(f"Added {ticker}")
    
    desc = truncate_description(info.get('longBusinessSummary', 'No description.'), 300)
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%); padding: 2rem; border-radius: 16px; color: white; margin-bottom: 2rem;'>
        <h2 style='margin: 0; font-size: 2.5rem; color: white;'>{info.get('longName', ticker)}</h2>
        <p style='margin: 0.5rem 0; font-size: 1.2rem; color: white;'>{ticker} | {info.get('exchange', 'N/A')}</p>
        <p style='margin: 1rem 0; color: white;'>{desc}</p>
    </div>
    """, unsafe_allow_html=True)
    
    price = data['Close'].iloc[-1]
    change = ((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100
    
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Price", f"${price:,.2f}", f"{change:+.2f}%")
    c2.metric("Market Cap", f"${info.get('marketCap', 0)/1e9:,.1f}B")
    c3.metric("P/E", f"{info.get('trailingPE', 0):,.2f}")
    c4.metric("Trading Volume", f"{int(data['Volume'].iloc[-1]):,}")
    c5.metric("52W Range", f"${info.get('fiftyTwoWeekLow', 0):,.0f}-{info.get('fiftyTwoWeekHigh', 0):,.0f}")
    
    st.markdown(f"""
    <div class='confidence-card'>
        <div style='display: grid; grid-template-columns: 1fr 2fr; gap: 2rem;'>
            <div style='text-align: center;'>
                <div style='font-size: 4rem; font-weight: 800; color: #3b82f6;'>{confidence}%</div>
                <div style='font-size: 0.9rem; color: #6b7280; text-transform: uppercase; margin-bottom: 1rem;'>Confidence Score</div>
                <div class='recommendation-badge {rec_class}'>{rec}</div>
            </div>
            <div>
                <p style='font-size: 1.25rem; line-height: 1.8; color: #111827; font-weight: 500;'>
                    Based on comprehensive analysis across four distinct analytical models—Valuation, Momentum, Earnings, and Technical—this stock receives a <strong>{confidence}%</strong> confidence score with a <strong>{rec}</strong> recommendation. Each model contributes weighted inputs (Valuation 30%, Momentum 25%, Earnings 25%, Technical 20%) that are synthesized into this overall assessment. The confidence score reflects the degree of alignment across these independent analytical approaches, with higher scores indicating strong consensus and lower scores suggesting conflicting signals that warrant caution.
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Detailed Analysis Section
    detailed_analysis = generate_detailed_analysis(confidence, rec, val_score, mom_score, earn_score, tech_score,
                                                   val_details, mom_details, earn_details, tech_details, info, data, ticker)
    
    st.markdown(f"""
    <div style='background: white; padding: 2.5rem; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); 
                margin: 2rem 0; border: 1px solid #e5e7eb;'>
        <h2 style='font-size: 2rem; font-weight: 700; color: #111827; margin-bottom: 1.5rem;'>Comprehensive Investment Analysis</h2>
        <div style='font-size: 1.05rem; line-height: 1.9; color: #374151;'>
            {detailed_analysis}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h2 class='section-header'>Investment Analysis Models</h2>", unsafe_allow_html=True)
    
    models = [
        ("Valuation Model", val_score, val_details),
        ("Momentum Model", mom_score, mom_details),
        ("Earnings Model", earn_score, earn_details),
        ("Technical Model", tech_score, tech_details)
    ]
    
    for title, score, dets in models:
        st.markdown(f"""
        <div class='model-card'>
            <div class='model-header'>{title}</div>
            <span class='model-score'>Score: {score+6}/12</span>
            <div class='model-description'>
                {' '.join(dets)}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Disclaimer Footer
    st.markdown("""
    <div class='disclaimer-footer'>
        <div class='disclaimer-text'>
            <strong>Disclaimer:</strong> The information provided by BARBAS is for informational and educational purposes only and should not be construed as financial, investment, or legal advice. 
            All analysis, recommendations, and confidence scores are generated using automated models and historical data, which may not accurately predict future performance. 
            Past performance is not indicative of future results. Investing in stocks and securities involves risk, including the potential loss of principal. 
            You should consult with a qualified financial advisor, accountant, or other professional before making any investment decisions. 
            BARBAS and its creators assume no responsibility or liability for any financial losses or damages resulting from the use of this tool or reliance on its outputs.
        </div>
    </div>
    """, unsafe_allow_html=True)

elif st.session_state.page == 'portfolio':
    st.markdown("<div class='brbas-header'><h1 class='brbas-title' style='color: #ffffff !important;'>BARBAS</h1></div>", unsafe_allow_html=True)
    st.markdown("<h2 class='section-header'>Portfolio</h2>", unsafe_allow_html=True)
    
    if not st.session_state.portfolio:
        st.info("Portfolio empty. Add stocks from Analysis page.")
    else:
        for t in st.session_state.portfolio:
            d, i, _ = get_stock_data(t, "1mo")
            if d is not None and not d.empty:
                price = d['Close'].iloc[-1]
                change = ((d['Close'].iloc[-1] / d['Close'].iloc[0]) - 1) * 100
                
                st.markdown(f"""
                <div class='portfolio-card' style='display: block;'>
                    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;'>
                        <div>
                            <h3 style='margin: 0; color: #111827; font-size: 1.8rem;'>{t}</h3>
                            <p style='margin: 0.25rem 0; color: #6b7280; font-size: 0.95rem;'>{i.get('longName', t)}</p>
                        </div>
                        <div style='text-align: right;'>
                            <div style='font-size: 2rem; font-weight: 800; color: #111827;'>${price:.2f}</div>
                            <div style='font-size: 1rem; color: {"#10b981" if change >= 0 else "#ef4444"}; font-weight: 600;'>
                                {change:+.2f}%
                            </div>
                        </div>
                    </div>
                    <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; padding-top: 1rem; border-top: 1px solid #e5e7eb;'>
                        <div>
                            <div style='font-size: 0.75rem; color: #6b7280; font-weight: 600;'>MARKET CAP</div>
                            <div style='font-size: 1.1rem; font-weight: 700; color: #111827;'>${i.get('marketCap', 0)/1e9:.1f}B</div>
                        </div>
                        <div>
                            <div style='font-size: 0.75rem; color: #6b7280; font-weight: 600;'>P/E RATIO</div>
                            <div style='font-size: 1.1rem; font-weight: 700; color: #111827;'>{i.get('trailingPE', 0):.2f}</div>
                        </div>
                        <div>
                            <div style='font-size: 0.75rem; color: #6b7280; font-weight: 600;'>TRADING VOLUME</div>
                            <div style='font-size: 1.1rem; font-weight: 700; color: #111827;'>{int(d['Volume'].iloc[-1]):,}</div>
                        </div>
                        <div>
                            <div style='font-size: 0.75rem; color: #6b7280; font-weight: 600;'>52W RANGE</div>
                            <div style='font-size: 1.1rem; font-weight: 700; color: #111827;'>${i.get('fiftyTwoWeekLow', 0):.0f}-{i.get('fiftyTwoWeekHigh', 0):.0f}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("Analyze Full Report", key=f"analyze_{t}", use_container_width=True):
                    st.session_state.page = 'analysis'
                    st.session_state.selected_ticker = t
                    st.rerun()
                
                st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)
    
    # Disclaimer Footer
    st.markdown("""
    <div class='disclaimer-footer'>
        <div class='disclaimer-text'>
            <strong>Disclaimer:</strong> The information provided by BARBAS is for informational and educational purposes only and should not be construed as financial, investment, or legal advice. 
            All analysis, recommendations, and confidence scores are generated using automated models and historical data, which may not accurately predict future performance. 
            Past performance is not indicative of future results. Investing in stocks and securities involves risk, including the potential loss of principal. 
            You should consult with a qualified financial advisor, accountant, or other professional before making any investment decisions. 
            BARBAS and its creators assume no responsibility or liability for any financial losses or damages resulting from the use of this tool or reliance on its outputs.
        </div>
    </div>
    """, unsafe_allow_html=True)

elif st.session_state.page == 'top_stocks':
    st.markdown("<div class='brbas-header'><h1 class='brbas-title' style='color: #ffffff !important;'>BARBAS</h1></div>", unsafe_allow_html=True)
    st.markdown("<h2 class='section-header'>Top Stocks by Sector</h2>", unsafe_allow_html=True)
    
    sectors = {
        'Technology': ['AAPL', 'MSFT', 'GOOGL'],
        'Healthcare': ['JNJ', 'UNH', 'PFE'],
        'Financial': ['JPM', 'BAC', 'WFC'],
        'Consumer': ['AMZN', 'TSLA', 'HD'],
        'Energy': ['XOM', 'CVX', 'COP']
    }
    
    with st.spinner("Analyzing sectors..."):
        results = {}
        for sector, tickers in sectors.items():
            best_conf = 0
            best_stock = None
            for t in tickers:
                d, i, e = get_stock_data(t, "3mo")
                if e or d is None or d.empty:
                    continue
                for ma in [20, 50, 100, 200]:
                    d[f'MA{ma}'] = d['Close'].rolling(window=ma).mean()
                d = calculate_ema(d)
                d['RSI'] = calculate_rsi(d)
                d['MACD'], d['Signal'], d['Histogram'] = calculate_macd(d)
                
                vs, _ = analyze_valuation(i)
                ms, _ = analyze_momentum(d)
                es, _ = analyze_earnings(i)
                ts, _ = analyze_technical(d)
                
                conf = calculate_confidence_score(i, d, vs, ms, es, ts)
                if conf > best_conf:
                    best_conf = conf
                    rec, rc = get_recommendation_from_confidence(conf)
                    best_stock = {'ticker': t, 'name': i.get('longName', t), 'price': d['Close'].iloc[-1], 
                                'conf': conf, 'rec': rec, 'rc': rc}
            
            if best_stock:
                results[sector] = best_stock
    
    cols = st.columns(2)
    for idx, (sector, stock) in enumerate(results.items()):
        with cols[idx % 2]:
            color = '#10b981' if 'BUY' in stock['rec'] else '#ef4444' if 'SELL' in stock['rec'] else '#f59e0b'
            st.markdown(f"""
            <div style='background: white; padding: 1.5rem; border-radius: 12px; border: 1px solid #e5e7eb; 
                        box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 1rem;'>
                <div style='font-size: 0.85rem; color: #6b7280; text-transform: uppercase;'>{sector}</div>
                <h3 style='margin: 0.5rem 0; color: #111827;'>{stock['ticker']}</h3>
                <p style='color: #6b7280; font-size: 0.9rem;'>{stock['name'][:35]}</p>
                <div style='font-size: 1.5rem; font-weight: 700; color: #111827;'>${stock['price']:.2f}</div>
                <div style='background: {color}; color: white; padding: 0.4rem 1rem; border-radius: 20px; 
                            display: inline-block; margin-top: 0.5rem; font-weight: 600;'>{stock['rec']}</div>
                <div style='font-size: 1.2rem; font-weight: 700; color: #111827; margin-top: 0.5rem;'>
                    Confidence: {stock['conf']}%
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Analysis Summary
    st.markdown("<h2 class='section-header' style='margin-top: 3rem;'>Sector Analysis Summary</h2>", unsafe_allow_html=True)
    
    summary_text = f"""
    <div style='background: white; padding: 2.5rem; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); 
                margin: 2rem 0; border: 1px solid #e5e7eb;'>
        <p style='font-size: 1.05rem; line-height: 1.9; color: #374151; margin-bottom: 1.5rem;'>
            Our sector analysis methodology evaluates the top holdings within each major market sector to identify the most attractive opportunities based on our comprehensive four-model framework. For each sector, we analyzed {sum(len(tickers) for tickers in sectors.values())} leading stocks across {len(sectors)} major sectors, applying the same rigorous Valuation, Momentum, Earnings, and Technical analysis used in our individual stock assessments.
        </p>
        <p style='font-size: 1.05rem; line-height: 1.9; color: #374151; margin-bottom: 1.5rem;'>
            The stocks presented above represent the highest-confidence opportunities within their respective sectors—essentially the "best of breed" based on current market conditions and fundamental health. These selections change dynamically as market conditions evolve, with our models continuously re-evaluating sector leadership based on the latest data.
        </p>
        <p style='font-size: 1.05rem; line-height: 1.9; color: #374151;'>
            <strong>Interpretation Guide:</strong> Sectors showing multiple BUY recommendations suggest favorable industry dynamics and strong fundamental momentum. Conversely, sectors with HOLD or SELL recommendations may be facing headwinds that warrant caution. Diversified investors should consider maintaining exposure across multiple sectors while overweighting those showing the strongest conviction scores. Remember that sector rotation is a natural market phenomenon—today's laggards can become tomorrow's leaders, so periodic reassessment is crucial.
        </p>
    </div>
    """
    
    st.markdown(summary_text, unsafe_allow_html=True)
    
    # Disclaimer Footer
    st.markdown("""
    <div class='disclaimer-footer'>
        <div class='disclaimer-text'>
            <strong>Disclaimer:</strong> The information provided by BARBAS is for informational and educational purposes only and should not be construed as financial, investment, or legal advice. 
            All analysis, recommendations, and confidence scores are generated using automated models and historical data, which may not accurately predict future performance. 
            Past performance is not indicative of future results. Investing in stocks and securities involves risk, including the potential loss of principal. 
            You should consult with a qualified financial advisor, accountant, or other professional before making any investment decisions. 
            BARBAS and its creators assume no responsibility or liability for any financial losses or damages resulting from the use of this tool or reliance on its outputs.
        </div>
    </div>
    """, unsafe_allow_html=True)

elif st.session_state.page == 'compare':
    st.markdown("<div class='brbas-header'><h1 class='brbas-title' style='color: #ffffff !important;'>BARBAS</h1></div>", unsafe_allow_html=True)
    st.markdown("<h2 class='section-header'>Compare Stocks</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        ticker1 = search_ticker(st.text_input("First Stock", "AAPL", key="compare1"))
    with col2:
        ticker2 = search_ticker(st.text_input("Second Stock", "MSFT", key="compare2"))
    
    period = st.selectbox("Comparison Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=3, key="compare_period")
    
    if st.button("Compare Stocks", use_container_width=True):
        with st.spinner("Analyzing stocks..."):
            # Get data for both stocks
            data1, info1, error1 = get_stock_data(ticker1, period)
            data2, info2, error2 = get_stock_data(ticker2, period)
            
            if error1 or error2 or data1 is None or data2 is None or data1.empty or data2.empty:
                st.error("Unable to fetch data for one or both stocks")
            else:
                # Calculate indicators for both
                for data in [data1, data2]:
                    for ma in [20, 50, 100, 200]:
                        data[f'MA{ma}'] = data['Close'].rolling(window=ma).mean()
                    data = calculate_ema(data)
                    data['RSI'] = calculate_rsi(data)
                    data['MACD'], data['Signal'], data['Histogram'] = calculate_macd(data)
                
                # Analyze both stocks
                val_score1, val_details1 = analyze_valuation(info1)
                mom_score1, mom_details1 = analyze_momentum(data1)
                earn_score1, earn_details1 = analyze_earnings(info1)
                tech_score1, tech_details1 = analyze_technical(data1)
                conf1 = calculate_confidence_score(info1, data1, val_score1, mom_score1, earn_score1, tech_score1)
                rec1, rec_class1 = get_recommendation_from_confidence(conf1)
                
                val_score2, val_details2 = analyze_valuation(info2)
                mom_score2, mom_details2 = analyze_momentum(data2)
                earn_score2, earn_details2 = analyze_earnings(info2)
                tech_score2, tech_details2 = analyze_technical(data2)
                conf2 = calculate_confidence_score(info2, data2, val_score2, mom_score2, earn_score2, tech_score2)
                rec2, rec_class2 = get_recommendation_from_confidence(conf2)
                
                # Display comparison
                st.markdown("<h3 style='margin-top: 2rem; color: #111827;'>Confidence Comparison</h3>", unsafe_allow_html=True)
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"""
                    <div class='confidence-card'>
                        <h3 style='color: #111827; margin-bottom: 1rem;'>{ticker1} - {info1.get('longName', ticker1)}</h3>
                        <div style='text-align: center;'>
                            <div style='font-size: 3.5rem; font-weight: 800; color: #3b82f6;'>{conf1}%</div>
                            <div class='recommendation-badge {rec_class1}' style='margin-top: 1rem;'>{rec1}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_b:
                    st.markdown(f"""
                    <div class='confidence-card'>
                        <h3 style='color: #111827; margin-bottom: 1rem;'>{ticker2} - {info2.get('longName', ticker2)}</h3>
                        <div style='text-align: center;'>
                            <div style='font-size: 3.5rem; font-weight: 800; color: #3b82f6;'>{conf2}%</div>
                            <div class='recommendation-badge {rec_class2}' style='margin-top: 1rem;'>{rec2}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Model score comparison
                st.markdown("<h3 style='margin-top: 2rem; color: #111827;'>Model Score Comparison</h3>", unsafe_allow_html=True)
                
                comparison_data = {
                    'Model': ['Valuation', 'Momentum', 'Earnings', 'Technical'],
                    ticker1: [val_score1+6, mom_score1+6, earn_score1+6, tech_score1+6],
                    ticker2: [val_score2+6, mom_score2+6, earn_score2+6, tech_score2+6]
                }
                
                fig = go.Figure(data=[
                    go.Bar(name=ticker1, x=comparison_data['Model'], y=comparison_data[ticker1], marker_color='#3b82f6'),
                    go.Bar(name=ticker2, x=comparison_data['Model'], y=comparison_data[ticker2], marker_color='#10b981')
                ])
                fig.update_layout(
                    barmode='group',
                    title='Model Scores (out of 12)',
                    yaxis=dict(range=[0, 12]),
                    height=400,
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color='#111827')
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Key metrics comparison
                st.markdown("<h3 style='margin-top: 2rem; color: #111827;'>Key Metrics Comparison</h3>", unsafe_allow_html=True)
                
                col_c, col_d = st.columns(2)
                with col_c:
                    price1 = data1['Close'].iloc[-1]
                    change1 = ((data1['Close'].iloc[-1] / data1['Close'].iloc[0]) - 1) * 100
                    st.markdown(f"""
                    <div style='background: white; padding: 2rem; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border: 1px solid #e5e7eb;'>
                        <h4 style='color: #111827; margin-bottom: 1.5rem;'>{ticker1}</h4>
                        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;'>
                            <div><span style='color: #6b7280; font-size: 0.85rem;'>Price:</span><br><strong style='font-size: 1.3rem; color: #111827;'>${price1:.2f}</strong></div>
                            <div><span style='color: #6b7280; font-size: 0.85rem;'>Change:</span><br><strong style='font-size: 1.3rem; color: {"#10b981" if change1 >= 0 else "#ef4444"};'>{change1:+.2f}%</strong></div>
                            <div><span style='color: #6b7280; font-size: 0.85rem;'>Market Cap:</span><br><strong style='color: #111827;'>${info1.get('marketCap', 0)/1e9:.1f}B</strong></div>
                            <div><span style='color: #6b7280; font-size: 0.85rem;'>P/E:</span><br><strong style='color: #111827;'>{info1.get('trailingPE', 0):.2f}</strong></div>
                            <div><span style='color: #6b7280; font-size: 0.85rem;'>Trading Volume:</span><br><strong style='color: #111827;'>{int(data1['Volume'].iloc[-1]):,}</strong></div>
                            <div><span style='color: #6b7280; font-size: 0.85rem;'>Profit Margin:</span><br><strong style='color: #111827;'>{info1.get('profitMargins', 0)*100:.1f}%</strong></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_d:
                    price2 = data2['Close'].iloc[-1]
                    change2 = ((data2['Close'].iloc[-1] / data2['Close'].iloc[0]) - 1) * 100
                    st.markdown(f"""
                    <div style='background: white; padding: 2rem; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border: 1px solid #e5e7eb;'>
                        <h4 style='color: #111827; margin-bottom: 1.5rem;'>{ticker2}</h4>
                        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;'>
                            <div><span style='color: #6b7280; font-size: 0.85rem;'>Price:</span><br><strong style='font-size: 1.3rem; color: #111827;'>${price2:.2f}</strong></div>
                            <div><span style='color: #6b7280; font-size: 0.85rem;'>Change:</span><br><strong style='font-size: 1.3rem; color: {"#10b981" if change2 >= 0 else "#ef4444"};'>{change2:+.2f}%</strong></div>
                            <div><span style='color: #6b7280; font-size: 0.85rem;'>Market Cap:</span><br><strong style='color: #111827;'>${info2.get('marketCap', 0)/1e9:.1f}B</strong></div>
                            <div><span style='color: #6b7280; font-size: 0.85rem;'>P/E:</span><br><strong style='color: #111827;'>{info2.get('trailingPE', 0):.2f}</strong></div>
                            <div><span style='color: #6b7280; font-size: 0.85rem;'>Trading Volume:</span><br><strong style='color: #111827;'>{int(data2['Volume'].iloc[-1]):,}</strong></div>
                            <div><span style='color: #6b7280; font-size: 0.85rem;'>Profit Margin:</span><br><strong style='color: #111827;'>{info2.get('profitMargins', 0)*100:.1f}%</strong></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Comparison summary
                st.markdown("<h3 style='margin-top: 2rem; color: #111827;'>Comparison Summary</h3>", unsafe_allow_html=True)
                
                winner = ticker1 if conf1 > conf2 else ticker2 if conf2 > conf1 else "Tie"
                winner_conf = max(conf1, conf2)
                
                st.markdown(f"""
                <div style='background: white; padding: 2.5rem; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); 
                            margin: 2rem 0; border: 1px solid #e5e7eb;'>
                    <h4 style='color: #111827; margin-bottom: 1rem;'>Investment Verdict</h4>
                    <p style='font-size: 1.05rem; line-height: 1.8; color: #374151;'>
                        Based on our comprehensive four-model analysis, <strong>{winner}</strong> emerges as the stronger investment opportunity with a confidence score of {winner_conf}%. 
                        {ticker1} scored {conf1}% while {ticker2} scored {conf2}%, representing a {abs(conf1-conf2):.1f} percentage point difference.
                    </p>
                    <p style='font-size: 1.05rem; line-height: 1.8; color: #374151; margin-top: 1rem;'>
                        {'Both stocks show strong characteristics and could be suitable for a diversified portfolio.' if abs(conf1-conf2) < 10 else 
                         f'The {winner} demonstrates notably superior characteristics across multiple analytical dimensions.' if winner != "Tie" else
                         'Both stocks are evenly matched in terms of investment attractiveness.'}
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
    # Disclaimer Footer
    st.markdown("""
    <div class='disclaimer-footer'>
        <div class='disclaimer-text'>
            <strong>Disclaimer:</strong> The information provided by BARBAS is for informational and educational purposes only and should not be construed as financial, investment, or legal advice. 
            All analysis, recommendations, and confidence scores are generated using automated models and historical data, which may not accurately predict future performance. 
            Past performance is not indicative of future results. Investing in stocks and securities involves risk, including the potential loss of principal. 
            You should consult with a qualified financial advisor, accountant, or other professional before making any investment decisions. 
            BARBAS and its creators assume no responsibility or liability for any financial losses or damages resulting from the use of this tool or reliance on its outputs.
        </div>
    </div>
    """, unsafe_allow_html=True)import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="BARBAS", layout="wide", page_icon="📊", initial_sidebar_state="expanded")

# Professional CSS - Apple/Airbnb/Revolut inspired
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: #ffffff;
        padding: 0;
    }
    
    .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }
    
    .brbas-header {
        text-align: center;
        padding: 4rem 0;
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
        margin: -2rem -3rem 2rem -3rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    .brbas-title {
        font-size: 9rem !important;
        font-weight: 800 !important;
        color: #ffffff !important;
        letter-spacing: 2.5rem !important;
        margin: 0 !important;
        text-shadow: none !important;
    }
    
    .brbas-header h1 {
        color: #ffffff !important;
    }
    
    .sidebar-title {
        font-size: 2.8rem !important;
        font-weight: 800;
        letter-spacing: 0.8rem;
        text-align: center;
        margin-bottom: 2rem !important;
        color: white !important;
        text-shadow: none;
    }
    
    .model-card {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
        margin: 1.5rem 0;
        transition: box-shadow 0.2s ease;
    }
    
    .model-card:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    
    .model-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 1rem;
    }
    
    .model-score {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border-radius: 24px;
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 1rem;
    }
    
    .model-description {
        color: #374151;
        font-size: 1rem;
        line-height: 1.7;
        margin: 1rem 0 0 0;
    }
    
    .model-detail {
        padding: 0;
        margin: 0;
        font-size: 1rem;
        line-height: 1.7;
        color: #374151;
    }
    
    .section-header {
        font-size: 1.875rem;
        font-weight: 700;
        color: #111827;
        margin: 3rem 0 1.5rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #e5e7eb;
    }
    
    .confidence-card {
        background: white;
        padding: 2.5rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
        margin: 2rem 0;
    }
    
    .recommendation-badge {
        display: inline-block;
        padding: 0.75rem 2rem;
        border-radius: 24px;
        font-weight: 700;
        font-size: 1.1rem;
        margin: 1rem 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .strong-buy { background: #10b981; color: white; }
    .buy { background: #34d399; color: white; }
    .hold { background: #fbbf24; color: white; }
    .sell { background: #f87171; color: white; }
    .strong-sell { background: #ef4444; color: white; }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e40af 0%, #3b82f6 100%);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown h1 {
        color: white !important;
    }
    
    [data-testid="stSidebar"] button {
        width: 100%;
        background: rgba(255, 255, 255, 0.1);
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 0.875rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.2s ease;
    }
    
    [data-testid="stSidebar"] button:hover {
        background: rgba(255, 255, 255, 0.2);
        border-color: rgba(255, 255, 255, 0.3);
    }
    
    .stMetric {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
    }
    
    .stMetric label {
        color: #6b7280 !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: #111827 !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    .portfolio-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
        margin-bottom: 1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        transition: box-shadow 0.2s ease;
    }
    
    .portfolio-card:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    
    .disclaimer-footer {
        background: #f9fafb;
        padding: 2rem;
        border-radius: 12px;
        margin-top: 4rem;
        border-top: 1px solid #e5e7eb;
        text-align: center;
    }
    
    .disclaimer-text {
        font-size: 0.8rem;
        color: #6b7280;
        line-height: 1.6;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Dark text for all content */
    p, span, div {
        color: #374151 !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #111827 !important;
    }
    
    strong {
        color: #111827 !important;
    }
</style>
""", unsafe_allow_html=True)

# Session state
if 'page' not in st.session_state:
    st.session_state.page = 'analysis'
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = []

# Sidebar
with st.sidebar:
    st.markdown("<h1 class='sidebar-title'>BARBAS</h1>", unsafe_allow_html=True)
    if st.button("Stock Analysis"):
        st.session_state.page = 'analysis'
    if st.button("Compare Stocks"):
        st.session_state.page = 'compare'
    if st.button("Portfolio"):
        st.session_state.page = 'portfolio'
    if st.button("Top Stocks"):
        st.session_state.page = 'top_stocks'

def search_ticker(query):
    mapping = {
        'apple': 'AAPL', 'microsoft': 'MSFT', 'google': 'GOOGL', 'amazon': 'AMZN',
        'tesla': 'TSLA', 'meta': 'META', 'nvidia': 'NVDA'
    }
    return mapping.get(query.lower(), query.upper())

def truncate_description(text, max_length=300):
    if len(text) <= max_length:
        return text
    truncated = text[:max_length]
    last_period = truncated.rfind('.')
    if last_period > max_length * 0.7:
        return truncated[:last_period + 1]
    return truncated[:truncated.rfind(' ')] + '.'

def format_number(num):
    """Format numbers with commas for thousands"""
    if num is None or num == 0:
        return "0"
    if isinstance(num, (int, float)):
        return f"{num:,.0f}" if num >= 1 else f"{num:.2f}"
    return str(num)

def calculate_confidence_score(info, data, val_score, mom_score, earn_score, tech_score):
    weights = {'valuation': 0.30, 'momentum': 0.25, 'earnings': 0.25, 'technical': 0.20}
    val_norm = ((val_score + 6) / 12) * 100
    mom_norm = ((mom_score + 6) / 12) * 100
    earn_norm = ((earn_score + 6) / 12) * 100
    tech_norm = ((tech_score + 6) / 12) * 100
    return round(val_norm * weights['valuation'] + mom_norm * weights['momentum'] + 
                 earn_norm * weights['earnings'] + tech_norm * weights['technical'], 1)

def get_recommendation_from_confidence(confidence):
    if confidence >= 75: return "STRONG BUY", "strong-buy"
    elif confidence >= 60: return "BUY", "buy"
    elif confidence >= 40: return "HOLD", "hold"
    elif confidence >= 25: return "SELL", "sell"
    else: return "STRONG SELL", "strong-sell"

def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    return 100 - (100 / (1 + gain / loss))

def calculate_macd(data, fast=12, slow=26, signal=9):
    exp1 = data['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = data['Close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line, macd - signal_line

def calculate_ema(data, periods=[8, 21, 50]):
    for period in periods:
        data[f'EMA{period}'] = data['Close'].ewm(span=period, adjust=False).mean()
    return data

def generate_detailed_analysis(confidence, rec, val_score, mom_score, earn_score, tech_score, 
                               val_details, mom_details, earn_details, tech_details, info, data, ticker):
    """Generate investment strategy and risk analysis"""
    
    analysis_parts = []
    
    # Forward-looking recommendations
    analysis_parts.append("<strong style='font-size: 1.2rem; color: #111827;'>Investment Strategy & Outlook:</strong>")
    
    if confidence >= 75:
        analysis_parts.append(f"For investors seeking growth opportunities, consider building positions in {ticker} with a focus on dollar-cost averaging over the next 1-2 months to optimize entry points. Our models suggest potential upside of 15-25% over the next 6-12 months based on current trajectories. The strong confidence score reflects alignment across valuation metrics, momentum indicators, earnings performance, and technical signals—all pointing toward favorable conditions for capital appreciation.")
        analysis_parts.append(f"Key catalysts to monitor include upcoming earnings reports, sector rotation trends, and broader market sentiment shifts. Consider setting profit targets at key resistance levels while maintaining stop-losses 8-10% below entry to protect against unexpected volatility. This stock demonstrates the characteristics of a core portfolio holding with both near-term momentum and longer-term fundamental support. Position sizing should reflect your risk tolerance, but this represents a high-conviction opportunity where larger allocations may be warranted for growth-oriented investors.")
    elif confidence >= 60:
        analysis_parts.append(f"Investors may consider initiating or adding to positions in {ticker}, but should do so with measured position sizing and clear risk parameters. The stock appears well-positioned for modest appreciation over the next 6-12 months, with potential gains in the 8-15% range if fundamentals continue on their current trajectory. The positive but not overwhelming confidence score suggests good prospects balanced with some areas of caution that prevent a stronger recommendation.")
        analysis_parts.append(f"Monitor quarterly earnings closely and be prepared to adjust positions if key metrics deteriorate. This is a suitable holding for core portfolio positions with medium-term horizons. Consider a phased entry approach, deploying 50% of your intended position initially and adding the remainder if the stock confirms support at current levels. Set alerts for significant price movements and review your thesis quarterly to ensure the original investment case remains intact.")
    elif confidence >= 40:
        analysis_parts.append(f"For current shareholders of {ticker}, maintaining existing positions appears reasonable while monitoring for clearer directional signals. New investors should wait for better entry opportunities or stronger confirmation across our analytical models before initiating positions. The neutral confidence score reflects mixed signals—some positive indicators offset by concerns in other areas—suggesting the stock may trade within a relatively tight range near current levels over the coming months.")
        analysis_parts.append(f"Watch for potential catalysts that could shift the risk-reward profile—either positive developments that would upgrade the recommendation, or negative signals that would warrant position reduction. This is not an optimal time for aggressive accumulation, but patient investors may find opportunities if the stock pulls back to stronger support levels. Consider reducing position size if held in overweight positions, rebalancing to market-weight exposure until a clearer trend emerges.")
    elif confidence >= 25:
        analysis_parts.append(f"Investors holding {ticker} should consider reducing position sizes and reallocating capital to higher-conviction opportunities. The current risk-reward profile appears unfavorable, with meaningful downside potential if current trends persist. The below-average confidence score reflects concerns across multiple analytical dimensions that suggest the path of least resistance is lower in the near to medium term.")
        analysis_parts.append(f"For those maintaining exposure, implement strict stop-loss levels around 5-7% below current prices and avoid averaging down without clear evidence of trend reversal. New investors should avoid initiating positions until confidence metrics improve substantially. Monitor for potential bottoming patterns, but recognize that catching a falling knife is rarely profitable—better opportunities exist elsewhere in the current market environment.")
    else:
        analysis_parts.append(f"We recommend existing shareholders consider exiting positions in {ticker} or implementing protective strategies such as options hedging. The convergence of negative signals across our models suggests elevated risk of further downside, with the low confidence score reflecting serious concerns about both current valuation and future prospects. Capital preservation should be the priority given the unfavorable setup across multiple timeframes and analytical approaches.")
        analysis_parts.append(f"Investors may find better opportunities elsewhere in the current market environment where risk-reward profiles are more favorable. Only consider re-entry after observing sustained improvement across multiple quarters and clearer technical reversal patterns—specifically, look for positive earnings surprises, improving relative strength, and confirmation above key moving averages before revisiting this investment. The market is telling you something with this configuration, and fighting the tape is typically an expensive proposition.")
    
    # Risk considerations
    volatility = data['Close'].pct_change().std() * np.sqrt(252) * 100
    analysis_parts.append(f"<br><strong style='font-size: 1.2rem; color: #111827;'>Risk Profile:</strong>")
    
    if volatility > 30:
        analysis_parts.append(f"With annualized volatility of {volatility:.1f}%, {ticker} exhibits elevated price fluctuations that require careful position sizing and risk management. This level of volatility means the stock can easily move 2-3% in a single session, and 10-15% swings over a few weeks are not uncommon. Investors should size positions accordingly—typically no more than 2-3% of portfolio value for conservative investors, or up to 5% for those with higher risk tolerance.")
        analysis_parts.append(f"The elevated volatility creates both opportunity and risk. Nimble traders may find attractive entry and exit points during these swings, but buy-and-hold investors must have the temperament to withstand significant paper losses during drawdowns. Consider using options strategies like protective puts or covered calls to manage risk if holding larger positions, and ensure this stock fits within your overall portfolio's risk budget.")
    elif volatility > 20:
        analysis_parts.append(f"With annualized volatility of {volatility:.1f}%, {ticker} demonstrates moderate volatility suitable for most risk tolerances. This puts the stock roughly in line with broader market volatility, suggesting normal price behavior without excessive swings. Standard position sizing rules apply—generally 3-5% of portfolio value depending on conviction level and your overall risk tolerance.")
        analysis_parts.append(f"The moderate volatility profile makes this accessible to a wide range of investors, from conservative to aggressive. While you should still expect occasional 5-10% pullbacks as part of normal market action, the stock shouldn't produce the white-knuckle ride that higher-volatility names deliver. This makes it suitable for both trading and longer-term holding, depending on your investment approach and time horizon.")
    else:
        analysis_parts.append(f"With annualized volatility of {volatility:.1f}%, {ticker} shows relatively stable price action compared to the broader market. This lower volatility makes the stock appropriate for conservative investors and can allow for larger position sizes—potentially 5-7% of portfolio value—without exceeding reasonable risk parameters. The stock tends to move methodically rather than erratically, which can be both a blessing and a curse.")
        analysis_parts.append(f"While the stability is attractive for risk-averse investors, recognize that lower volatility also typically means more modest returns over shorter timeframes. This stock is unlikely to double quickly, but also less likely to be cut in half. It represents a core holding appropriate for the foundation of a portfolio, providing steady exposure without the drama of more volatile alternatives. Patient, long-term investors often prefer this profile.")
    
    return "<br><br>".join(analysis_parts)

def analyze_valuation(info):
    score = 0
    details = []
    
    methodology = "The Valuation Model employs a multi-metric approach to determine whether a stock is trading at an attractive price relative to its intrinsic value and growth prospects. We primarily analyze the PEG (Price/Earnings-to-Growth) ratio, which compares the P/E ratio to the company's earnings growth rate, providing crucial context that raw P/E ratios miss. A PEG below 1.0 typically indicates undervaluation—you're paying less than $1 for every percentage point of growth. We also examine the trailing P/E ratio against historical averages and sector comparables to identify deviation from normal valuation ranges. The model accounts for sector-specific dynamics, as technology companies naturally command higher multiples than utilities due to growth differentials."
    
    # Try multiple ways to get PEG ratio
    peg = info.get('pegRatio')
    if peg is None or peg <= 0:
        # Try to calculate PEG manually if we have P/E and growth rate
        pe = info.get('trailingPE') or info.get('forwardPE')
        growth = info.get('earningsQuarterlyGrowth') or info.get('earningsGrowth')
        if pe and growth and pe > 0 and growth > 0:
            peg = pe / (growth * 100)
    
    if peg and peg > 0:
        if peg < 1:
            score += 2
            details.append(f"PEG Ratio of {peg:.2f} indicates undervaluation. The stock trades below its growth rate, suggesting a compelling value opportunity.")
        elif peg < 2:
            score += 1
            details.append(f"PEG Ratio of {peg:.2f} suggests fair valuation. The stock is reasonably priced relative to its growth.")
        else:
            score -= 2
            details.append(f"PEG Ratio of {peg:.2f} indicates overvaluation. The stock is expensive relative to its growth prospects.")
    else:
        details.append("PEG Ratio data unavailable. This metric compares P/E to growth rate.")
    
    pe = info.get('trailingPE')
    if pe and pe > 0:
        if pe < 15:
            score += 1
            details.append(f"P/E Ratio of {pe:.1f} is below market average, suggesting potential undervaluation.")
        elif pe > 30:
            score -= 1
            details.append(f"P/E Ratio of {pe:.1f} is elevated, indicating premium pricing.")
    else:
        details.append("P/E Ratio unavailable.")
    
    evaluation = f"This model scored {score+6}/12. "
    if score >= 2:
        evaluation += "The strong positive score indicates the stock is trading at an attractive valuation relative to fundamentals, presenting a favorable entry point for value-conscious investors. Multiple metrics confirm pricing below intrinsic value estimates."
    elif score >= 0:
        evaluation += "The neutral score suggests fair valuation—the stock is neither obviously cheap nor expensive. Investors are paying a reasonable price for the underlying business, though significant margin of safety may be limited."
    else:
        evaluation += "The negative score signals overvaluation concerns. The stock appears expensive relative to fundamentals, suggesting limited upside and elevated risk if growth expectations aren't met. Valuation-sensitive investors may want to wait for better entry points."
    
    details.insert(0, methodology)
    details.append(evaluation)
    
    return score, details

def analyze_momentum(data):
    score = 0
    details = []
    
    methodology = "The Momentum Model evaluates trend strength and direction using moving average analysis, a cornerstone of technical analysis that has proven predictive power across all timeframes. We examine the relationship between short-term (20-50 day) and long-term (200-day) moving averages to identify the stock's position within its trend cycle. A 'Golden Cross'—when the 50-day MA crosses above the 200-day MA—is one of the most reliable bullish signals in technical analysis, often preceding sustained rallies. Conversely, a 'Death Cross' signals deteriorating momentum. We also measure the stock's deviation from its moving averages; excessive distance from the mean often leads to mean reversion moves. This model helps identify whether a stock is in accumulation, markup, distribution, or markdown phases."
    
    price = data['Close'].iloc[-1]
    
    if 'MA50' in data.columns and 'MA200' in data.columns:
        ma50 = data['MA50'].iloc[-1]
        ma200 = data['MA200'].iloc[-1]
        if pd.notna(ma50) and pd.notna(ma200):
            if ma50 > ma200 and price > ma50:
                score += 2
                details.append(f"Golden Cross active with price at ${price:.2f} above both 50-day (${ma50:.2f}) and 200-day (${ma200:.2f}) moving averages. This bullish configuration suggests strong upward momentum.")
            elif ma50 < ma200 and price < ma50:
                score -= 2
                details.append(f"Death Cross active with price at ${price:.2f} below both moving averages. This bearish pattern indicates downward pressure.")
            else:
                details.append(f"Mixed signals with price at ${price:.2f}. The stock is in transition between trends.")
        else:
            details.append("Insufficient data for moving average analysis.")
    else:
        details.append("Moving average data unavailable.")
    
    if 'MA20' in data.columns:
        ma20 = data['MA20'].iloc[-1]
        if pd.notna(ma20):
            deviation = ((price / ma20) - 1) * 100
            if abs(deviation) > 3:
                if deviation > 0:
                    score += 1
                    details.append(f"Strong short-term momentum with price {deviation:.1f}% above 20-day MA.")
                else:
                    score -= 1
                    details.append(f"Weak short-term momentum with price {abs(deviation):.1f}% below 20-day MA.")
    
    evaluation = f"This model scored {score+6}/12. "
    if score >= 2:
        evaluation += "The strong momentum score indicates the stock is in a confirmed uptrend with buyers in control. Price action above key moving averages suggests continued strength, making this favorable for trend-following strategies. Momentum tends to persist, so this bullish setup often continues until clear reversal signals emerge."
    elif score >= 0:
        evaluation += "The neutral momentum score suggests the stock is in consolidation or transition. Neither bulls nor bears have clear control, resulting in range-bound or choppy price action. Wait for clearer directional confirmation before making significant moves based on momentum alone."
    else:
        evaluation += "The negative momentum score indicates the stock is in a downtrend with sellers in control. Price action below key moving averages suggests continued weakness. Fighting bearish momentum is typically a losing proposition—better to wait for clear stabilization and reversal signals before considering entry."
    
    details.insert(0, methodology)
    details.append(evaluation)
    
    return score, details

def analyze_earnings(info):
    score = 0
    details = []
    
    methodology = "The Earnings Model focuses on the fundamental health and growth trajectory of the underlying business, examining the company's ability to generate and grow profits over time. We analyze quarterly earnings growth rates, which provide insight into business momentum and operational execution. Consistent earnings growth typically drives long-term stock appreciation, as market valuations ultimately reflect corporate profitability. We also evaluate profit margins, which reveal competitive positioning and pricing power—high-margin businesses can better weather economic downturns and have more flexibility for reinvestment. This model essentially asks: is this company making more money quarter after quarter, and how efficiently does it convert revenue into profit?"
    
    growth = info.get('earningsQuarterlyGrowth')
    if growth is not None:
        if growth > 0.15:
            score += 2
            details.append(f"Exceptional earnings growth of {growth*100:.1f}% significantly exceeds market averages, demonstrating strong operational performance.")
        elif growth > 0.05:
            score += 1
            details.append(f"Solid earnings growth of {growth*100:.1f}% indicates healthy business expansion.")
        elif growth > 0:
            details.append(f"Modest earnings growth of {growth*100:.1f}% shows steady but slow expansion.")
        else:
            score -= 2
            details.append(f"Earnings decline of {growth*100:.1f}% is concerning and suggests operational challenges.")
    else:
        details.append("Earnings growth data unavailable.")
    
    margin = info.get('profitMargins')
    if margin:
        if margin > 0.20:
            details.append(f"Strong profit margin of {margin*100:.1f}% demonstrates pricing power and operational efficiency.")
        elif margin > 0.10:
            details.append(f"Decent profit margin of {margin*100:.1f}% indicates reasonable profitability.")
        else:
            details.append(f"Thin profit margin of {margin*100:.1f}% suggests limited pricing power.")
    
    evaluation = f"This model scored {score+6}/12. "
    if score >= 2:
        evaluation += "The strong earnings score indicates robust fundamental health with accelerating profitability. Companies demonstrating this level of earnings power typically see their stocks appreciate over time as the market recognizes and rewards sustainable growth. This fundamental strength provides a solid foundation for stock price appreciation."
    elif score >= 0:
        evaluation += "The neutral earnings score suggests stable but unspectacular fundamental performance. The company is maintaining profitability but not demonstrating the dynamic growth that drives significant stock appreciation. This is acceptable for mature, dividend-paying companies but less exciting for growth investors."
    else:
        evaluation += "The negative earnings score raises red flags about fundamental business health. Declining earnings often precede stock price weakness as the market discounts deteriorating fundamentals. Until earnings trends stabilize and reverse, the fundamental case for ownership remains challenged regardless of other factors."
    
    details.insert(0, methodology)
    details.append(evaluation)
    
    return score, details

def analyze_technical(data):
    score = 0
    details = []
    
    methodology = "The Technical Model employs momentum oscillators and price-action indicators to identify overbought and oversold conditions that often precede trend reversals or continuations. The Relative Strength Index (RSI) measures the magnitude of recent price changes to evaluate whether a stock has moved too far, too fast in either direction. RSI readings below 30 typically indicate oversold conditions where selling pressure has been exhausted, often creating buying opportunities. Readings above 70 suggest overbought conditions where a pullback becomes probable. We also analyze the MACD (Moving Average Convergence Divergence), which tracks the relationship between two exponential moving averages to identify changes in momentum. MACD crossovers above the signal line generate buy signals, while crossovers below generate sell signals."
    
    rsi = data['RSI'].iloc[-1] if 'RSI' in data.columns and pd.notna(data['RSI'].iloc[-1]) else None
    if rsi:
        if rsi < 30:
            score += 2
            details.append(f"RSI at {rsi:.1f} indicates oversold conditions. Historically, readings below 30 precede price reversals as selling pressure becomes exhausted.")
        elif rsi > 70:
            score -= 2
            details.append(f"RSI at {rsi:.1f} signals overbought conditions. Readings above 70 typically lead to pullbacks as buyers become exhausted.")
        else:
            details.append(f"RSI at {rsi:.1f} is in neutral territory, indicating balanced buying and selling pressure.")
    else:
        details.append("RSI data unavailable.")
    
    if 'MACD' in data.columns and 'Signal' in data.columns:
        macd = data['MACD'].iloc[-1]
        signal = data['Signal'].iloc[-1]
        if pd.notna(macd) and pd.notna(signal):
            if macd > signal:
                score += 1
                details.append(f"MACD bullish at {macd:.2f} above signal line {signal:.2f}, indicating upward momentum.")
            else:
                score -= 1
                details.append(f"MACD bearish at {macd:.2f} below signal line {signal:.2f}, suggesting downward pressure.")
    
    evaluation = f"This model scored {score+6}/12. "
    if score >= 2:
        evaluation += "The strong
