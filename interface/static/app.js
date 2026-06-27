const API = "/api";
const BIKE_COLOR = "#2563eb";
const DOCK_COLOR = "#f97316";

let stations = [];
let map = null;
let markers = {};
let currentStation = null;
let dayPredictions = null;
let activeTab = "prediction";
let dayChart = null;

function todayDate() {
  const now = new Date();
  const offset = now.getTimezoneOffset() * 60000;
  return new Date(now - offset).toISOString().slice(0, 10);
}

async function loadStations() {
  const response = await fetch(`${API}/stations`);
  stations = await response.json();
  populateSelect();
  initializeMap();
}

function populateSelect() {
  const select = document.getElementById("select-station");
  select.innerHTML =
    '<option value="">— Choisir une station —</option>' +
    stations
      .slice()
      .sort((a, b) => a.station_name.localeCompare(b.station_name))
      .map((s) => `<option value="${s.station}">${s.station_name} (#${s.station})</option>`)
      .join("");

  select.addEventListener("change", () => {
    if (select.value) selectStation(parseInt(select.value, 10));
  });

  document.getElementById("recherche-station").addEventListener("input", (e) => {
    const term = e.target.value.toLowerCase();
    for (const option of select.options) {
      if (!option.value) continue;
      option.hidden = !option.textContent.toLowerCase().includes(term);
    }
  });
}

function initializeMap() {
  map = L.map("carte").setView([43.6047, 1.4442], 13);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "&copy; OpenStreetMap contributors",
    maxZoom: 19,
  }).addTo(map);

  stations.forEach((s) => {
    const marker = L.marker([s.lat, s.lon]).addTo(map);
    marker.bindPopup(`<b>${s.station_name}</b><br>Capacité : ${s.capacity} places`);
    marker.on("click", () => selectStation(s.station));
    markers[s.station] = marker;
  });
}

async function selectStation(stationId) {
  currentStation = stations.find((s) => s.station === stationId);
  if (!currentStation) return;

  document.getElementById("select-station").value = stationId;
  document.getElementById("panneau").classList.remove("cache");
  document.getElementById("nom-station").textContent = currentStation.station_name;
  document.getElementById("info-station").textContent =
    `Station n°${currentStation.station} — capacité ${currentStation.capacity} places`;

  map.panTo([currentStation.lat, currentStation.lon]);
  markers[stationId].openPopup();

  await loadPrediction();
}

async function loadPrediction() {
  const dateInput = document.getElementById("date-prediction");
  if (!dateInput.value) dateInput.value = todayDate();

  const response = await fetch(
    `${API}/predict_day?station=${currentStation.station}&date=${dateInput.value}`
  );
  if (!response.ok) return;
  dayPredictions = await response.json();

  updateHourResult();
  if (activeTab === "stats") drawDayChart();
}

const STATE_LABELS = { safe: "Tranquille", uncertain: "Incertain", high_risk: "Risque élevé" };

function updateHourResult() {
  const hour = parseInt(document.getElementById("heure-prediction").value, 10);
  document.getElementById("heure-affichee").textContent = hour;
  if (!dayPredictions) return;

  const point = dayPredictions.predictions[hour];
  const bikeRisk = point.bike_shortage_risk;
  const dockRisk = point.dock_shortage_risk;

  document.getElementById("resultat-prediction").innerHTML = `
    <div class="carte-risque ${bikeRisk.state}">
      <div class="carte-risque-icone">🚲</div>
      <div class="carte-risque-titre">Vélo</div>
      <div class="carte-risque-etat">${STATE_LABELS[bikeRisk.state]}</div>
    </div>
    <div class="carte-risque ${dockRisk.state}">
      <div class="carte-risque-icone">🅿️</div>
      <div class="carte-risque-titre">Place</div>
      <div class="carte-risque-etat">${STATE_LABELS[dockRisk.state]}</div>
    </div>
  `;
}

function activateTab(name) {
  activeTab = name;
  document.querySelectorAll(".onglet-bouton").forEach((b) => b.classList.toggle("actif", b.dataset.onglet === name));
  document.querySelectorAll(".onglet-contenu").forEach((c) => c.classList.toggle("cache", c.id !== `onglet-${name}`));

  if (name === "stats" && dayPredictions) drawDayChart();
}

document.querySelectorAll(".onglet-bouton").forEach((btn) => {
  btn.addEventListener("click", () => activateTab(btn.dataset.onglet));
});

function drawDayChart() {
  const hours = dayPredictions.predictions.map((p) => `${p.hour}h`);
  const bikes = dayPredictions.predictions.map((p) => p.predicted_available_bikes);
  const docks = dayPredictions.predictions.map((p) => p.predicted_available_bike_stands);

  if (dayChart) dayChart.destroy();
  dayChart = new Chart(document.getElementById("graphique-jour"), {
    type: "line",
    data: {
      labels: hours,
      datasets: [
        { label: "Vélos disponibles (prédit)", data: bikes, borderColor: BIKE_COLOR, tension: 0.3 },
        { label: "Places libres (prédit)", data: docks, borderColor: DOCK_COLOR, tension: 0.3 },
      ],
    },
    options: { responsive: true, scales: { y: { beginAtZero: true } } },
  });
}

document.getElementById("heure-prediction").addEventListener("input", updateHourResult);
document.getElementById("date-prediction").addEventListener("change", loadPrediction);
document.getElementById("heure-prediction").value = new Date().getHours();

loadStations();
