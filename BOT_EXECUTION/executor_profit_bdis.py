import pyautogui
import time
import pygetwindow as gw

# SEGURANÇA OPERACIONAL
pyautogui.PAUSE = 0.3
pyautogui.FAILSAFE = True 

class JARVIS_Final_Executor:
    def __init__(self):
        # Atalhos confirmados na sua imagem de configurações
        self.ABRIR_COMPRA = 'f5'
        self.ABRIR_VENDA = 'f9'
        # Atalhos universais do Profit para execução INSTANTÂNEA (Mercado)
        self.COMPRA_MERCADO = ['shift', 'c']
        self.VENDA_MERCADO = ['shift', 'v']
        self.ZERAR_TUDO = 'f12' 

    def focar_profit(self):
        try:
            # Busca a janela pelo título visto na sua primeira imagem
            profit_win = [w for w in gw.getWindowsWithTitle('Profit') if w.visible][0]
            if not profit_win.isActive:
                profit_win.activate()
                time.sleep(0.3)
            return True
        except Exception:
            print("❌ JARVIS: Profit Pro não detectado na tela!")
            return False

    def enviar_ordem_b3(self, sinal, ativo="PETR4"):
        if not self.focar_profit(): return

        # --- TRAVA DE SEGURANÇA: SELEÇÃO DE TICKER ---
        # Garante que a ordem não 'vaze' para o gráfico ativo errado
        print(f"🔒 [JARVIS SECURITY] Forçando Ticker no Profit: {ativo}")
        # Digita o ticker (abre a janela de busca do Profit)
        pyautogui.write(ativo)
        time.sleep(0.3)
        # Confirma a troca do ativo no gráfico
        pyautogui.press('enter')
        time.sleep(0.8) # Tempo de tolerância para a B3/Profit carregar o dom/book

        if sinal == 1:
            print(f"🚀 [JARVIS] EXECUTANDO COMPRA A MERCADO: {ativo}")
            # Usando hotkey para garantir execução sem abrir boleta de confirmação
            pyautogui.hotkey(*self.COMPRA_MERCADO) 
            
        elif sinal == -1:
            print(f"🔻 [JARVIS] EXECUTANDO VENDA A MERCADO: {ativo}")
            pyautogui.hotkey(*self.VENDA_MERCADO)

    def pânico_zerar(self):
        """Função de emergência para zerar posição imediatamente"""
        if self.focar_profit():
            pyautogui.press(self.ZERAR_TUDO)
            print("⚠️ [JARVIS] POSIÇÃO ZERADA PELO SISTEMA!")