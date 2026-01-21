package com.scouting.ui;

import com.scouting.service.StatFilterCriteria;
import com.vaadin.flow.component.button.Button;
import com.vaadin.flow.component.button.ButtonVariant;
import com.vaadin.flow.component.combobox.ComboBox;
import com.vaadin.flow.component.html.Span;
import com.vaadin.flow.component.icon.Icon;
import com.vaadin.flow.component.icon.VaadinIcon;
import com.vaadin.flow.component.orderedlayout.HorizontalLayout;
import com.vaadin.flow.component.textfield.TextField;
import com.vaadin.flow.data.value.ValueChangeMode;
import java.util.Map;
import java.util.function.Consumer;

/**
 * Questo componente custom dimostra il principio di "Composizione" nella UI.
 * Invece di duplicare codice per ogni riga di filtro dinamico, abbiamo creato un blocco riutilizzabile
 * che contiene un selettore (ComboBox), due campi numerici (Min/Max) e un bottone di rimozione.
 * La classe gestisce autonomamente la propria logica grafica e notifica la vista principale tramite delle "Callback"
 * (interfacce funzionali) quando i dati cambiano o quando la riga deve essere eliminata.
 */

public class FilterRowComponent extends HorizontalLayout {
    
    private final ComboBox<String> statSelector;
    private final TextField minStatValue;
    private final TextField maxStatValue;
    private final Button removeBtn;
    
    private final Map<String, String> statFieldMapping;

    public FilterRowComponent(Map<String, String> mapping, 
                              Runnable onUpdateCallback, 
                              Consumer<FilterRowComponent> onDeleteCallback) {
        
        this.statFieldMapping = mapping;
        
        setAlignItems(Alignment.BASELINE);
        setSpacing(true);
        setPadding(false);
        setWidthFull();

        getStyle().set("min-height", "40px");

        statSelector = new ComboBox<>();
        statSelector.setItems(mapping.keySet());
        statSelector.setPlaceholder("Select Statistic...");
        statSelector.setWidth("220px");
        statSelector.setClearButtonVisible(true);

        minStatValue = new TextField();
        minStatValue.setPlaceholder("Min");
        minStatValue.setWidth("90px");
        minStatValue.setValueChangeMode(ValueChangeMode.LAZY);

        maxStatValue = new TextField();
        maxStatValue.setPlaceholder("Max");
        maxStatValue.setWidth("90px");
        maxStatValue.setValueChangeMode(ValueChangeMode.LAZY);

        removeBtn = new Button(new Icon(VaadinIcon.TRASH));
        removeBtn.addThemeVariants(ButtonVariant.LUMO_ICON, ButtonVariant.LUMO_ERROR);
        removeBtn.setTooltipText("Remove this filter");

       
        removeBtn.addClickListener(e -> onDeleteCallback.accept(this));

        statSelector.addValueChangeListener(e -> onUpdateCallback.run());
        minStatValue.addValueChangeListener(e -> onUpdateCallback.run());
        maxStatValue.addValueChangeListener(e -> onUpdateCallback.run());

        add(statSelector, new Span("from"), minStatValue, new Span("to"), maxStatValue, removeBtn);
    }


    public StatFilterCriteria getCriteria() {
        String selectedLabel = statSelector.getValue();
        if (selectedLabel == null || selectedLabel.isEmpty()) return null;

        String fieldName = statFieldMapping.get(selectedLabel);
        Double min = parseDoubleSafe(minStatValue.getValue());
        Double max = parseDoubleSafe(maxStatValue.getValue());

        if (min == null && max == null) return null;

        return new StatFilterCriteria(fieldName, min, max);
    }

    private Double parseDoubleSafe(String value) {
        if (value == null || value.trim().isEmpty()) return null;
        try {
            return Double.parseDouble(value.replace(",", "."));
        } catch (NumberFormatException e) {
            return null;
        }
    }
}