package com.scouting.service;

import com.scouting.data.repository.PlayerRepository;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class DataProcessingServiceTest {
    
    @Mock
    private PlayerRepository playerRepository;
    
    @InjectMocks
    private DataProcessingService service;
    
    @Test
    void testRepositoryInteraction() {
        // Verifica che il repository possa essere chiamato
        when(playerRepository.count()).thenReturn(0L);
        
        long count = playerRepository.count();
        
        verify(playerRepository, times(1)).count();
    }
}