import MetaTrader5 as mt5

MT5_LOGIN = 54206952
MT5_PASS = "jNQ4EM6#"
MT5_SERVER = "XPMT5-DEMO"

def detect_config():
    if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASS, server=MT5_SERVER):
        print(f"Erro ao iniciar: {mt5.last_error()}")
        return
    
    ticker = "BBDC4"
    # Tenta sem .SA primeiro (Padrão Simulador XP)
    if not mt5.symbol_select(ticker, True):
        # Tenta com .SA
        ticker = ticker + ".SA"
        if not mt5.symbol_select(ticker, True):
            print("Ativo BBDC4 não encontrado em nenhuma variação.")
            mt5.shutdown()
            return

    si = mt5.symbol_info(ticker)
    if si:
        print(f"\n--- CONFIGURAÇÃO DETECTADA PARA {ticker} ---")
        print(f"Símbolo Correto: {ticker}")
        print(f"Volume Min: {si.volume_min}")
        print(f"Volume Step: {si.volume_step}")
        
        # Filling Modes
        filling = si.type_filling
        modes = []
        if filling & mt5.SYMBOL_FILLING_FOK: modes.append("FOK (Fill or Kill)")
        if filling & mt5.SYMBOL_FILLING_IOC: modes.append("IOC (Immediate or Cancel)")
        if filling & mt5.SYMBOL_FILLING_BOC: modes.append("BOC (Book or Cancel)")
        
        print(f"Modos de Preenchimento Suportados: {', '.join(modes) if modes else 'Específicos da Corretora'}")
        
    mt5.shutdown()

if __name__ == "__main__":
    detect_config()
