(() => {
  'use strict';
  const $ = (selector, root = document) => root.querySelector(selector);
  const escapeText = (value) => String(value ?? '—');
  const state = { charge: null, store: null };
  const connection = $('#connection');
  const toast = $('#toast');
  let toastTimer;

  function showToast(message, kind = 'success') {
    toast.textContent = message;
    toast.className = `toast ${kind}`;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.add('hidden'), 4500);
  }
  async function api(path, options = {}) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 10000);
    try {
      const response = await fetch(path, { ...options, signal: controller.signal, headers: { ...(options.body ? { 'Content-Type': 'application/json' } : {}), ...options.headers } });
      const text = await response.text();
      let payload = null;
      try { payload = text ? JSON.parse(text) : null; } catch { throw new Error('The server returned an unreadable response.'); }
      if (!response.ok) throw new Error(payload?.error || payload?.message || `Request failed (${response.status}).`);
      return payload;
    } catch (error) {
      if (error.name === 'AbortError') throw new Error('The request timed out. Please try again.');
      if (error instanceof TypeError) throw new Error('Could not connect to the SmartHub server.');
      throw error;
    } finally { clearTimeout(timer); }
  }
  function setBusy(form, busy) {
    const button = $('button[type="submit"]', form);
    if (!button) return;
    if (busy) { button.dataset.original = button.textContent; button.disabled = true; button.textContent = 'Please wait…'; }
    else { button.disabled = false; button.textContent = button.dataset.original || button.textContent; }
  }
  function formData(form) { return Object.fromEntries(new FormData(form).entries()); }
  function renderError(container, message) {
    container.replaceChildren();
    const p = document.createElement('p'); p.className = 'empty-state error-text'; p.textContent = message; container.append(p);
  }
  function metric(key, value) { const el = $(`[data-metric="${key}"]`); if (el) el.textContent = value == null ? '—' : String(value); }
  const num = (v) => Number.isFinite(Number(v)) ? Number(v).toLocaleString() : '—';

  async function loadOverview() {
    const alert = $('#page-alert'); alert.classList.add('hidden');
    try {
      const [summary, health, events, prediction, usage, system] = await Promise.allSettled([
        api('/api/dashboard/summary'), api('/api/health'), api('/api/events?limit=8'), api('/api/ml/usage-prediction'), api('/api/metrics/usage?limit=6'), api('/api/status')
      ]);
      if (summary.status === 'fulfilled') {
        const data = summary.value;
        ['total_sessions', 'active_sessions', 'available_items', 'borrowed_items'].forEach(k => metric(k, data[k]));
        const counts = $('#service-counts'); counts.replaceChildren();
        const serviceCounts = data.service_counts || {};
        const keys = Object.keys(serviceCounts);
        if (!keys.length) { counts.innerHTML = '<p class="empty-state">No service activity recorded yet.</p>'; }
        else keys.forEach(name => {
          const row = document.createElement('div'); row.className = 'service-row';
          const label = document.createElement('span'); label.textContent = name;
          const value = document.createElement('strong'); value.textContent = num(serviceCounts[name]);
          const track = document.createElement('div'); track.className = 'bar-track'; const bar = document.createElement('i'); bar.style.width = `${Math.min(100, (Number(serviceCounts[name]) / Math.max(1, ...Object.values(serviceCounts).map(Number))) * 100)}%`; track.append(bar);
          row.append(label, track, value); counts.append(row);
        });
      } else { ['total_sessions','active_sessions','available_items','borrowed_items'].forEach(k => metric(k, null)); renderError($('#service-counts'), 'Service metrics are unavailable.'); }
      const status = $('#system-status'); status.replaceChildren();
      if (health.status === 'fulfilled' && health.value?.status === 'ok') {
        connection.className = 'status-pill online'; connection.lastChild.textContent = ' Connected';
        const p = document.createElement('p'); p.className = 'health-line'; p.textContent = 'SmartHub API is responding'; status.append(p);
        const d = document.createElement('p'); d.className = 'health-detail'; d.textContent = system.status === 'fulfilled' && system.value?.status === 'ok' ? `Database responding · ${num(system.value.summary?.active_sessions)} active sessions` : 'System summary is unavailable.'; status.append(d);
      } else {
        connection.className = 'status-pill offline'; connection.lastChild.textContent = ' Connection issue';
        const p = document.createElement('p'); p.className = 'error-text'; p.textContent = 'Health service is unavailable.'; status.append(p);
      }
      const predictionEl = $('#prediction');
      if (prediction.status === 'fulfilled' && prediction.value.available && prediction.value.predicted_demand != null) {
        const d = prediction.value; predictionEl.textContent = `${num(d.predicted_demand)} predicted CHARGE sessions in the next hour${d.confidence != null ? ` · ${Math.round(d.confidence * 100)}% confidence` : ''}${d.model ? ` · ${d.model}` : ''}`;
      } else predictionEl.textContent = prediction.status === 'fulfilled' ? (prediction.value.error || 'Forecast not configured.') : 'Prediction service unavailable.';
      const list = $('#events'); list.replaceChildren();
      if (events.status !== 'fulfilled') renderError(list, 'Recent activity is unavailable.');
      else if (!events.value.length) list.innerHTML = '<p class="empty-state">No events have been recorded yet.</p>';
      else events.value.forEach(event => {
        const row = document.createElement('div'); row.className = 'event-row';
        const icon = document.createElement('span'); icon.className = 'event-dot'; icon.textContent = (event.service || 'S').slice(0,1);
        const detail = document.createElement('div'); const title = document.createElement('strong'); title.textContent = event.event_type || 'System event'; const desc = document.createElement('p'); desc.textContent = event.message || event.service || 'Activity recorded'; detail.append(title, desc);
        const time = document.createElement('time'); time.textContent = event.created_at || '';
        row.append(icon, detail, time); list.append(row);
      });
      const history = $('#usage-history'); history.replaceChildren();
      if (usage.status !== 'fulfilled') renderError(history, 'Usage history is unavailable.');
      else if (!usage.value.length) history.innerHTML = '<p class="empty-state">No completed or closed sessions yet.</p>';
      else usage.value.forEach(item => {
        const row = document.createElement('div'); row.className = 'event-row';
        const icon = document.createElement('span'); icon.className = 'event-dot'; icon.textContent = (item.service_name || 'S').slice(0, 1);
        const detail = document.createElement('div'); const title = document.createElement('strong'); title.textContent = `${item.service_name || 'Service'} session #${item.id}`; const desc = document.createElement('p'); desc.textContent = `${item.status || 'closed'}${item.duration_seconds != null ? ` · ${num(item.duration_seconds)} seconds` : ''}`; detail.append(title, desc);
        const time = document.createElement('time'); time.textContent = item.started_at || '';
        row.append(icon, detail, time); history.append(row);
      });
    } catch (error) { alert.textContent = error.message; alert.className = 'alert error'; }
  }

  function renderChargeSession(result, stopping = false) {
    const box = $('#charge-session'); box.replaceChildren();
    const badge = document.createElement('span'); badge.className = `session-badge ${result.status === 'active' ? 'active' : ''}`; badge.textContent = result.status || (stopping ? 'completed' : '');
    const title = document.createElement('p'); title.className = 'empty-title'; title.textContent = `Session #${result.session_id}`;
    const detail = document.createElement('p'); detail.className = 'muted'; detail.textContent = result.duration_seconds != null ? `Duration: ${result.duration_seconds} seconds` : `User #${result.user_id ?? '—'} · Transaction #${result.transaction_id ?? '—'}`;
    box.append(badge, title, detail);
    if (result.status === 'active') {
      const button = document.createElement('button'); button.type = 'button'; button.className = 'button danger'; button.textContent = 'Stop charging';
      button.addEventListener('click', async () => {
        if (!window.confirm(`Stop charging session #${result.session_id}?`)) return;
        button.disabled = true;
        try { const stopped = await api('/api/charge/stop', { method: 'POST', body: JSON.stringify({ session_id: result.session_id }) }); state.charge = null; renderChargeSession(stopped, true); showToast('Charging session stopped.'); }
        catch (error) { button.disabled = false; showToast(error.message, 'error'); }
      }); box.append(button);
    }
  }
  $('#charge-start-form').addEventListener('submit', async event => {
    event.preventDefault(); const form = event.currentTarget; const data = formData(form); data.amount = Number(data.amount);
    if (!data.rfid_uid.trim() || !Number.isFinite(data.amount) || data.amount < 0 || data.amount > 10000) { showToast('Enter a valid RFID ID and amount from 0 to 10,000.', 'error'); return; }
    if (state.charge) { showToast('A charging session is already active in this page.', 'error'); return; }
    setBusy(form, true);
    try { const result = await api('/api/charge/start', { method: 'POST', body: JSON.stringify(data) }); state.charge = result; renderChargeSession(result); showToast(`Charging session #${result.session_id} started.`); }
    catch (error) { showToast(error.message, 'error'); }
    finally { setBusy(form, false); }
  });

  function renderStoreSession(result) {
    const box = $('#store-session'); box.replaceChildren();
    const badge = document.createElement('span'); badge.className = `session-badge ${result.status === 'active' ? 'active' : ''}`; badge.textContent = result.status || 'active';
    const title = document.createElement('p'); title.className = 'empty-title'; title.textContent = `Session #${result.session_id}`;
    const detail = document.createElement('p'); detail.className = 'muted'; detail.textContent = result.locked === false ? 'Locker unlocked. Complete the action at the kiosk.' : result.duration_seconds != null ? `Duration: ${result.duration_seconds} seconds` : `User #${result.user_id ?? '—'}`;
    box.append(badge, title, detail);
    if (result.status === 'active') {
      const actions = document.createElement('div'); actions.className = 'button-row';
      const unlock = document.createElement('button'); unlock.type = 'button'; unlock.className = 'button primary'; unlock.textContent = result.locked === false ? 'Unlocked' : 'Unlock locker'; unlock.disabled = result.locked === false;
      unlock.addEventListener('click', async () => { if (!window.confirm(`Unlock the locker for session #${result.session_id}? Make sure you are at the kiosk.`)) return; unlock.disabled = true; try { const value = await api('/api/store/unlock', { method: 'POST', body: JSON.stringify({ session_id: result.session_id }) }); renderStoreSession({ ...result, ...value, status: 'active' }); showToast('The backend confirmed the locker unlocked.'); } catch (error) { unlock.disabled = false; showToast(error.message, 'error'); } });
      const finish = document.createElement('button'); finish.type = 'button'; finish.className = 'button secondary'; finish.textContent = 'Finish session';
      finish.addEventListener('click', async () => { if (!window.confirm(`Finish storage session #${result.session_id}? The locker will be locked.`)) return; finish.disabled = true; try { const value = await api('/api/store/finish', { method: 'POST', body: JSON.stringify({ session_id: result.session_id }) }); state.store = null; renderStoreSession(value); showToast(value.status === 'completed' ? 'Storage session completed.' : `Session ended with status: ${value.status}.`, value.status === 'completed' ? 'success' : 'error'); } catch (error) { finish.disabled = false; showToast(error.message, 'error'); } });
      actions.append(unlock, finish); box.append(actions);
    }
  }
  $('#store-start-form').addEventListener('submit', async event => {
    event.preventDefault(); const form = event.currentTarget; const data = formData(form);
    if (!data.rfid_uid.trim()) { showToast('Enter your RFID ID.', 'error'); return; }
    if (state.store) { showToast('A storage session is already active in this page.', 'error'); return; }
    setBusy(form, true); try { const result = await api('/api/store/start', { method: 'POST', body: JSON.stringify(data) }); state.store = result; renderStoreSession(result); showToast(`Storage session #${result.session_id} started.`); }
    catch (error) { showToast(error.message, 'error'); } finally { setBusy(form, false); }
  });

  async function loadItems() {
    const list = $('#items-list'); list.replaceChildren();
    try {
      const items = await api('/api/borrow/items');
      if (!items.length) { list.innerHTML = '<p class="empty-state">No items are currently available to borrow.</p>'; return; }
      items.forEach(item => {
        const row = document.createElement('div'); row.className = 'item-row'; const icon = document.createElement('span'); icon.className = 'item-icon'; icon.textContent = '▱';
        const info = document.createElement('div'); const name = document.createElement('strong'); name.textContent = item.name || item.item_type || 'Shared item'; const id = document.createElement('small'); id.textContent = `RFID: ${item.rfid_uid || item.id || '—'}`; info.append(name, id);
        const badge = document.createElement('span'); badge.className = 'availability'; badge.textContent = item.status || 'available'; row.append(icon, info, badge);
        row.addEventListener('click', () => { $('#borrow-item').value = item.rfid_uid || ''; $('#borrow-item').focus(); }); list.append(row);
      });
    } catch (error) { renderError(list, error.message); }
  }
  $('#borrow-form').addEventListener('submit', async event => {
    event.preventDefault(); const form = event.currentTarget; const data = formData(form);
    if (!data.rfid_uid.trim() || !data.item_rfid.trim()) { showToast('Enter both RFID IDs.', 'error'); return; }
    setBusy(form, true); try { const result = await api('/api/borrow/start', { method: 'POST', body: JSON.stringify(data) }); showToast(`Item borrowed. Record #${result.borrow_record_id}; vision check ${result.vision_result?.available ? 'available' : 'not configured'}.`); form.reset(); await loadItems(); }
    catch (error) { showToast(error.message, 'error'); } finally { setBusy(form, false); }
  });
  $('#return-form').addEventListener('submit', async event => {
    event.preventDefault(); const form = event.currentTarget; const data = formData(form);
    if (!data.item_rfid.trim()) { showToast('Enter the item RFID ID.', 'error'); return; }
    if (!window.confirm(`Submit return for item ${data.item_rfid}?`)) return;
    setBusy(form, true); try { const result = await api('/api/borrow/return', { method: 'POST', body: JSON.stringify(data) }); showToast(result.status === 'returned' ? 'Return confirmed by SmartHub.' : 'Return response received.', result.status === 'returned' ? 'success' : 'error'); form.reset(); await loadItems(); }
    catch (error) { showToast(error.message, 'error'); } finally { setBusy(form, false); }
  });

  function navigate() {
    const page = (location.hash || '#overview').slice(1); const target = document.getElementById(`page-${page}`) ? page : 'overview';
    document.querySelectorAll('.page').forEach(el => el.classList.toggle('active', el.id === `page-${target}`));
    document.querySelectorAll('.nav-link').forEach(el => { const active = el.dataset.page === target; el.classList.toggle('active', active); if (active) el.setAttribute('aria-current', 'page'); else el.removeAttribute('aria-current'); });
    if (target === 'borrow') loadItems();
    if (target === 'overview') loadOverview();
  }
  window.addEventListener('hashchange', navigate);
  $('#refresh-button').addEventListener('click', () => { loadOverview(); if (location.hash === '#borrow') loadItems(); });
  document.querySelectorAll('[data-refresh]').forEach(el => el.addEventListener('click', loadOverview));
  $('#items-refresh').addEventListener('click', loadItems);
  navigate();
})();
