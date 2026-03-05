import MetaTrader5 as mt5

MT5_LOGIN = 54206952
MT5_PASS = "And1and1411208*#"
MT5_SERVER = "XPMT5-DEMO"

def check_volume():
    if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASS, server=MT5_SERVER):
        print("Falha ao iniciar")
        return
    
    symbol = "BBDC4"
    mt5.symbol_select(symbol, True)
    si = mt5.symbol_info(symbol)
    if si:
        print(f"Propriedades de Volume para {symbol}:")
        print(f"Volume Min: {si.volume_min}")
        print(f"Volume Max: {si.volume_max}")
        print(f"Volume Step: {si.volume_step}")
        print(f"Trade Mode: {si.trade_mode}")
        print(f"Execution Mode: {si.trade_exemode}")
        print(f"Filling Modes Bitmask: {si.type_filling}")
    else:
        print("Ativo não encontrado")
    
    mt5.shutdown()

if __name__ == "__main__":
    check_volume()
