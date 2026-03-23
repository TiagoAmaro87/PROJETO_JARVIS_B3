"""
JARVIS_B3 - MT5 Executor (replaces PyAutoGUI)
Direct API execution via MetaTrader 5 - no screen clicking.
"""
import os
import time
from datetime import datetime, timezone
from typing import Optional
from loguru import logger

try:
    import MetaTrader5 as mt5
    HAS_MT5 = True
except ImportError:
    HAS_MT5 = False


class B3Executor:
    """Executes trades on B3 via MetaTrader 5 API."""

    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self._connected = False
        self._order_count = 0
        self._positions = {}

        if dry_run:
            logger.info("[B3-EXEC] DRY_RUN mode")
        else:
            logger.info("[B3-EXEC] LIVE mode - orders will be sent to MT5")

    def connect(self, path: str = None, login: int = None,
                password: str = None, server: str = None) -> bool:
        if not HAS_MT5:
            logger.error("[B3-EXEC] MetaTrader5 package not installed")
            return False

        kwargs = {}
        if path:
            kwargs["path"] = path
        if login:
            kwargs["login"] = login
        if password:
            kwargs["password"] = password
        if server:
            kwargs["server"] = server

        if not mt5.initialize(**kwargs):
            logger.error(f"[B3-EXEC] MT5 init failed: {mt5.last_error()}")
            return False

        account = mt5.account_info()
        logger.info(f"[B3-EXEC] Connected: {account.login} | "
                    f"Balance: R${account.balance:,.2f} | Leverage: 1:{account.leverage}")
        self._connected = True
        return True

    def get_current_symbol(self, symbol: str) -> str:
        """Find the correct tradable symbol (e.g., WINJ24 for WIN$)."""
        if not self._connected: return symbol
        
        # 1. Try raw
        if mt5.symbol_select(symbol, True): return symbol
        
        # 2. Try with .SA (standard for some setups)
        if mt5.symbol_select(symbol + ".SA", True): return symbol + ".SA"
        
        # 3. Special Mini Handling: WIN$ -> current contract
        year = str(datetime.now().year)[2:] # "26"
        if "WIN" in symbol:
            for letter in ["J", "M", "Q", "V", "Z", "G"]: # Common months
                test = f"WIN{letter}{year}"
                if mt5.symbol_select(test, True): return test
        
        if "WDO" in symbol:
            for letter in ["F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z"]:
                test = f"WDO{letter}{year}"
                if mt5.symbol_select(test, True): return test

        # 4. Fractional fallback (e.g., MULT3F)
        if mt5.symbol_select(symbol + "F", True): return symbol + "F"

        return symbol

    def disconnect(self):
        if self._connected and HAS_MT5:
            mt5.shutdown()
            self._connected = False

    def get_tick(self, symbol: str) -> Optional[dict]:
        if not self._connected:
            return None
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return None
        return {"bid": tick.bid, "ask": tick.ask, "last": tick.last, "volume": tick.volume}

    def comprar(self, symbol: str, lote: float, stop_loss: float = 0,
                take_profit: float = 0, comment: str = "JARVIS_B3") -> dict:
        """Buy order on B3."""
        return self._send_order(symbol, "buy", lote, stop_loss, take_profit, comment)

    def vender(self, symbol: str, lote: float, stop_loss: float = 0,
               take_profit: float = 0, comment: str = "JARVIS_B3") -> dict:
        """Sell order on B3."""
        return self._send_order(symbol, "sell", lote, stop_loss, take_profit, comment)

    def zerar_posicao(self, symbol: str) -> dict:
        """Close all positions for a symbol."""
        if self.dry_run:
            logger.info(f"[B3-EXEC] DRY CLOSE ALL {symbol}")
            return {"status": "dry_closed"}

        if not self._connected:
            return {"error": "not connected"}

        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            return {"status": "no_position"}

        results = []
        for pos in positions:
            result = self._close_position(pos)
            results.append(result)

        return {"closed": len(results), "results": results}

    def get_posicoes_abertas(self) -> list[dict]:
        if not self._connected:
            return []
        positions = mt5.positions_get()
        if not positions:
            return []
        return [{
            "ticket": p.ticket,
            "symbol": p.symbol,
            "type": "buy" if p.type == 0 else "sell",
            "volume": p.volume,
            "price_open": p.price_open,
            "price_current": p.price_current,
            "profit": p.profit,
            "sl": p.sl,
            "tp": p.tp,
        } for p in positions]

    def _send_order(self, symbol: str, side: str, lote: float,
                    sl: float, tp: float, comment: str) -> dict:
        self._order_count += 1

        if self.dry_run:
            logger.info(f"[B3-EXEC] DRY {side.upper()} {lote} {symbol} SL={sl} TP={tp}")
            return {
                "id": f"B3-DRY-{self._order_count:06d}",
                "symbol": symbol, "side": side, "volume": lote,
                "sl": sl, "tp": tp, "status": "dry_filled",
            }

        if not self._connected:
            return {"error": "not connected"}

        # Ensure symbol is selected and resolve current contract/fractional
        symbol = self.get_current_symbol(symbol)
        if not mt5.symbol_select(symbol, True):
            return {"error": f"symbol {symbol} not found in MT5"}

        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return {"error": f"no tick for {symbol} - check your broker connection"}

        order_type = mt5.ORDER_TYPE_BUY if side == "buy" else mt5.ORDER_TYPE_SELL
        price = tick.ask if side == "buy" else tick.bid

        # Detect filling mode
        sym_info = mt5.symbol_info(symbol)
        filling = mt5.ORDER_FILLING_FOK
        if sym_info:
            if sym_info.filling_mode & 2:
                filling = mt5.ORDER_FILLING_IOC
            elif sym_info.filling_mode & 4:
                filling = mt5.ORDER_FILLING_RETURN

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lote,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 789456,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling,
        }

        result = mt5.order_send(request)
        if result is None:
            return {"error": f"order_send failed: {mt5.last_error()}"}
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return {"error": f"rejected: {result.retcode} - {result.comment}"}

        logger.info(f"[B3-EXEC] {side.upper()} {lote} {symbol} @ {price:.2f} ticket={result.order}")
        return {
            "ticket": result.order, "symbol": symbol, "side": side,
            "volume": lote, "price": price, "sl": sl, "tp": tp, "status": "filled",
        }

    def _close_position(self, position) -> dict:
        close_type = mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(position.symbol)
        price = tick.bid if position.type == 0 else tick.ask

        sym_info = mt5.symbol_info(position.symbol)
        filling = mt5.ORDER_FILLING_FOK
        if sym_info and sym_info.filling_mode & 2:
            filling = mt5.ORDER_FILLING_IOC

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": close_type,
            "position": position.ticket,
            "price": price,
            "deviation": 20,
            "magic": 789456,
            "comment": "JARVIS_B3_CLOSE",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling,
        }

        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"[B3-EXEC] CLOSED {position.symbol} PnL={position.profit:.2f}")
            return {"ticket": position.ticket, "pnl": position.profit}
        return {"error": f"close failed: {mt5.last_error()}"}

    def modificar_stop(self, ticket: int, new_sl: float, new_tp: float = 0) -> bool:
        """Modify an existing position's SL/TP."""
        if self.dry_run:
            logger.info(f"[B3-EXEC] DRY MODIFY ticket={ticket} SL={new_sl}")
            return True

        if not self._connected: return False

        # Get existing position to keep TP if not provided
        pos = mt5.positions_get(ticket=ticket)
        if not pos or len(pos) == 0: return False
        pos = pos[0]
        
        tp = new_tp if new_tp > 0 else pos.tp

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": pos.symbol,
            "position": ticket,
            "sl": new_sl,
            "tp": tp,
        }

        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"[B3-EXEC] MODIFIED ticket={ticket} NEW SL={new_sl:.2f}")
            return True
        logger.error(f"[B3-EXEC] FAILED MODIFY ticket={ticket}: {mt5.last_error()}")
        return False
