import MetaTrader5 as mt5

MT5_LOGIN = 54206952
MT5_PASS = "And1and1411208*#"
MT5_SERVER = "XPMT5-DEMO"

def detect_config():
    if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASS, server=MT5_SERVER):
        print(f"Erro ao iniciar: {mt5.last_error()}")
        return
    
    ticker = "BBDC4"
    mt5.symbol_select(ticker, True)
    si = mt5.symbol_info(ticker)
    
    if si:
        print(f"\n--- DETALHES TÉCNICOS: {ticker} ---")
        # Mostrando todos os atributos disponíveis para não errar o nome
        for attr in dir(si):
            if not attr.startswith("_"):
                try:
                    print(f"{attr}: {getattr(si, attr)}")
                except:
                    pass
    
    mt5.shutdown()

if __name__ == "__main__":
    detect_config()
