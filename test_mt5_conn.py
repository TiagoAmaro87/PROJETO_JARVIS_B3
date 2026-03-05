import MetaTrader5 as mt5
import sys

MT5_LOGIN = 54206952
MT5_PASS = "And1and1411208*#"
MT5_SERVERS = ["XPMT5-DEMO", "XP-MT5-Demo", "PRD-MT5-XP"] # Tentando variações comuns

def test_conn():
    for server in MT5_SERVERS:
        print(f"--- Tentando Servidor: {server} ---")
        if mt5.initialize(login=MT5_LOGIN, password=MT5_PASS, server=server):
            print("✅ CONEXÃO ESTABELECIDA COM SUCESSO!")
            account_info = mt5.account_info()._asdict()
            print(f"Nome da Conta: {account_info.get('name')}")
            print(f"Saldo: {account_info.get('balance')}")
            return
        else:
            print(f"❌ Falha no servidor {server}: {mt5.last_error()}")
    
    print("\n[ERRO FINAL] Não foi possível conectar com nenhuma variação de servidor.")
    mt5.shutdown()

if __name__ == "__main__":
    test_conn()
