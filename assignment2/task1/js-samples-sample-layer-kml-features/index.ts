/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

function initMap(): void {
  const map = new google.maps.Map(
    document.getElementById("map") as HTMLElement,
    {
      zoom: 1,
      center: { lat: 0, lng: 0 },
    }
  );

  const kmlLayer = new google.maps.KmlLayer({
    url: "https://pastebin.com/raw/wwmSKgWN",
    suppressInfoWindows: true,
    map: map,
  });

  // for some reason ignored...
  map.setZoom(8);

  kmlLayer.addListener("click", (kmlEvent) => {
    const text = kmlEvent.featureData.description;

    showInContentWindow(text);
  });

  function showInContentWindow(text: string) {
    const sidebar = document.getElementById("sidebar") as HTMLElement;

    sidebar.innerHTML = text;
  }
}

declare global {
  interface Window {
    initMap: () => void;
  }
}
window.initMap = initMap;
export {};
